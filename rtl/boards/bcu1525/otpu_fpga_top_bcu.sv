// FPGA top for the BCU1525 (VCU1525 build: xcvu9p-fsgd2104-2L-e; docs/bcu1525.md): the block
// design otpu_bd (XDMA Gen3, the core clock, the SYSMON, the control interconnect and XDMA's DMA
// split onto the channels: boards/bcu1525/vivado/bd.tcl), one DDR4 LiteDRAM core per DIMM
// (otpu_ld4 at DDR4-1600, its own VexRiscv calibrating it at reset: boards/bcu1525/litedram_udimm/,
// USPDDRPHY for x8 UDIMMs without ECC, the lab card's; with OTPU_BCU_RDIMM defined
// boards/bcu1525/litedram/, x4 RDIMMs with ECC, the card's original modules) and otpu_bcu_sys:
// the accelerator (otpu_board, S = 2) and XDMA's DMA, meeting in one otpu_mem_ch per channel.
//
// Channels: board channel j = 2s + c (slice s's channel c; the host's channel_map) is DIMM
// dimm(j) = C0, C1, C3, C2 for j = 0..3: slice 0 on C0 (SLR0) and C1 (SLR1), slice 1 on C3 (SLR2)
// and C2 (SLR1), so each slice has one channel in its own SLR and one a crossing away
// (constraints/bcu1525_floorplan.xdc). The DIMMs' CSRs keep their DIMM's number (BAR0 0x40000 +
// d * 0x10000), so a channel's controller is found by its pins.
//
// Clocks: each DIMM's 300 MHz clock (clk300_<d>, the bank's LVDS input) on one global buffer feeds
// that DIMM's core (its MMCM makes sys_clk, 200 MHz, and the PHY's 800 MHz and IDELAYCTRL's 400
// MHz); clk300_1 (SLR1, with the PCIe block) also feeds the block design's MMCM: core_clk (the
// control side: the governor, the CSR gates, the SmartConnect's masters) and acc_clk, the same
// clock behind a clock-enable buffer, the accelerator's (no core_clk2x: bd.tcl refuses CLK2X, so
// TMEM_PMP is 0). xdma_aclk is XDMA's (250 MHz). otpu_mem_ch crosses between acc_clk, xdma_aclk
// and its DIMM's sys_clk.
// Power (the card has no fan: docs/bcu1525.md section 14): the thermal governor otpu_bcu_gov (BAR0
// 0x10000) decides what runs, as in the memory-test image (otpu_fpga_top_bcu_mem). After
// configuration everything is off (SAFE): the accelerator's clock stopped and its domain in reset,
// every DIMM's core in reset (its MMCM stopped, its DIMM unclocked and not refreshed: it
// calibrates again each time it is turned on). The accelerator runs only while the governor grants
// it and every DIMM, each up (the sequencer below: on, the clock and then ACC_RST_CYC cycles of
// reset; off, the reset at once and the clock ACC_STOP_CYC cycles later, inside the governor's
// OFF_DLY, so that otpu_mem_ch's holds see its reads in flight back while the controllers run).
// Its control registers are behind otpu_axil_off (OKAY, reads 0xDEAD0FF0, while it is not up:
// XDMA would turn an error into a completer abort) and otpu_axil_iso; each DIMM's CSRs are as in
// the memory image (off while the DIMM is off and UP_CYC cycles after it is turned on). Each
// channel's XDMA DMA is behind otpu_axi_off: no new burst unless the channel's DIMM is granted
// (dimm_ok), up and calibrated, the gate answering every burst (SLVERR) once the DIMM is off, with
// the channel's otpu_mem_ch XDMA side held in reset meanwhile (its reset handshake runs through
// the controller's clock, stopped then).
// Resets: core_rst from the block design (MMCM lock, PCIe PERST#), xrst from XDMA's axi_aresetn.
// The control SmartConnect (sc_ctl: XDMA's AXI-Lite master and a JTAG-to-AXI one) is reset with
// core_rstn, not with XDMA's reset, so the governor, the CSRs and the control registers answer
// over JTAG with the PCIe link down; core_rst is their masters' reset (otpu_axil_iso's m_rst).
// Each core's CSR port runs on core_clk, reset with its sys reset brought into core_clk (as the
// YPCB top: the two sides of its crossing reset together), behind otpu_axil_iso.
// The DIMM pins the single-rank core does not drive: par 0 (the RCD's parity check is off after
// reset; a UDIMM's DRAMs have C/A parity off, MR5), the other ranks' chip selects high, cke1 /
// odt1 low. The UART is DIMM C<UART_SEL>'s core's (the governor's CTRL; C1's after
// configuration): its BIOS log, the calibration. The board management
// controller's UART (bmc_tx / bmc_rx) is otpu_bmc_uart's (BAR0 0x20000, CAPS2 bit1): its watch
// reads the VCCINT regulator's temperatures for the governor (reg_hot, reg_age: HAS_REG) and
// lends the line to the host (opentpu/host/bmc.py: otpu-smi, otpu-bmc).
// The parameters are the target's (targets/bcu1525.toml: opentpu.target rtl_params; run_vivado.sh
// passes them, the defaults are its values: tests/test_target.py).
// the DIMMs' data pads and the cores' beat address bits (64-byte beats: 16 / 8 GiB a DIMM)
`ifdef OTPU_BCU_RDIMM
`define OTPU_BCU_DQ  72
`define OTPU_BCU_DQS 18
`define OTPU_BCU_AW  28
`define OTPU_BCU_LD_DM(N)
`else
`define OTPU_BCU_DQ  64
`define OTPU_BCU_DQS 8
`define OTPU_BCU_AW  27
`define OTPU_BCU_LD_DM(N) .ddram_dm(ddr4_c``N``_dm),
`endif
module otpu_fpga_top_bcu #(
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
  parameter int TMEM_IMPL  = 0,             // TMEM: 0 a copy per read port, 3 one copy
  parameter bit TMEM_RQ    = 1'b0,          //   (docs/tmem_nocopy.md; create_project.tcl TMEM_X,
  parameter int TMEM_NSB   = 8,             //   otpu_slice): RAMs a bank, the sub-bank's address
  parameter int TMEM_SBP   = 3,             //   bit, its hash, replays, true dual-port RAMs, QUANT
  parameter int TMEM_SBH   = 0,             //   3 and COLL the DMA's guests, the RAMs pumped at 2x
  parameter bit TMEM_RPL   = 1'b0,          //   (1: on core_clk2x, below), the read ports' ranks
  parameter bit TMEM_TDP   = 1'b0,          //   and RAM sides
  parameter bit TMEM_GST   = 1'b0,
  parameter int TMEM_PMP   = 0,
  parameter int TMEM_PRK   = 0,
  parameter int TMEM_SDM   = -1,
  parameter int MXU_IMPL   = 2,
  parameter int MXU_CL     = 16,
  parameter int VPU_CL     = 2,
  parameter int ULANES     = 8,
  parameter int TRACE_DEPTH = 16384,
  parameter int PQ_WIN     = 1024,
  parameter bit CHASH      = 1'b1,
  parameter bit DSTEP      = 1'b1,
  parameter int CORE_KHZ   = 100000,        // core_clk as the block design makes it
  parameter logic [31:0] BUILD_ID = 32'h0,  // the git commit (BUILD_ID register)
  parameter int DDR_MTS    = 1200,          // the DDR4 data rate the cores run (DDR_MTS register)
  parameter bit SEQ_WC     = 1'b0,    // EXPERIMENTAL: otpu_seq's wait entries (WAITC, docs/seq_vpu.md; CFG_UARCH bit24)
  parameter int PLATFORM   = 2,             // targets/bcu1525.toml (CFG_PLATFORM)
  parameter logic [31:0] ARCH_HASH = 32'h0, // its arch_hash (CFG_HASH; run_vivado.sh)
  parameter int PCIE_LANES = 16,            // Gen3 x16 (or x8: bd.tcl)
  parameter int UP_CYC     = 1 << 17        // a DIMM's CSRs and DMA answered here after turning on
) (
  // ---- board
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
  inout  wire         i2c_scl,
  inout  wire         i2c_sda,
  output logic        i2c_mux_rst_n,
  output logic        bmc_tx,               // the board management controller's UART (BB19 / BA19)
  input  logic        bmc_rx,
  // ---- PCIe Gen3 (the GT pins are placed by XDMA's select_quad: GTY quads 224-227)
  input  logic        pcie_refclk_p,
  input  logic        pcie_refclk_n,
  input  logic        pcie_perst_n,
  input  logic [PCIE_LANES-1:0] pcie_mgt_rxp,
  input  logic [PCIE_LANES-1:0] pcie_mgt_rxn,
  output logic [PCIE_LANES-1:0] pcie_mgt_txp,
  output logic [PCIE_LANES-1:0] pcie_mgt_txn,
  // ---- the four DDR4 DIMMs (C0 .. C3; constraints/ddr4_c<d>.udimm.pins.xdc, with OTPU_BCU_RDIMM
  // ddr4_c<d>.pins.xdc; par, cke1, odt1 and cs_hi: the pins the core does not drive; dm: the
  // UDIMM's DM_n, on the pins an RDIMM has its upper nibbles' strobes)
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
`ifndef OTPU_BCU_RDIMM
  output logic [7:0]  ddr4_c0_dm,
`endif
  inout  wire  [`OTPU_BCU_DQ-1:0]  ddr4_c0_dq,
  inout  wire  [`OTPU_BCU_DQS-1:0] ddr4_c0_dqs_p,
  inout  wire  [`OTPU_BCU_DQS-1:0] ddr4_c0_dqs_n,
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
`ifndef OTPU_BCU_RDIMM
  output logic [7:0]  ddr4_c1_dm,
`endif
  inout  wire  [`OTPU_BCU_DQ-1:0]  ddr4_c1_dq,
  inout  wire  [`OTPU_BCU_DQS-1:0] ddr4_c1_dqs_p,
  inout  wire  [`OTPU_BCU_DQS-1:0] ddr4_c1_dqs_n,
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
`ifndef OTPU_BCU_RDIMM
  output logic [7:0]  ddr4_c2_dm,
`endif
  inout  wire  [`OTPU_BCU_DQ-1:0]  ddr4_c2_dq,
  inout  wire  [`OTPU_BCU_DQS-1:0] ddr4_c2_dqs_p,
  inout  wire  [`OTPU_BCU_DQS-1:0] ddr4_c2_dqs_n,
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
`ifndef OTPU_BCU_RDIMM
  output logic [7:0]  ddr4_c3_dm,
`endif
  inout  wire  [`OTPU_BCU_DQ-1:0]  ddr4_c3_dq,
  inout  wire  [`OTPU_BCU_DQS-1:0] ddr4_c3_dqs_p,
  inout  wire  [`OTPU_BCU_DQS-1:0] ddr4_c3_dqs_n
);
  localparam int NC = 2 * S;
  // board channel j's DIMM (see the header)
  function automatic int dimm(int j);
    return j == 2 ? 3 : j == 3 ? 2 : j;
  endfunction
`ifndef SYNTHESIS
  initial if (S != 2) $fatal(1, "otpu_fpga_top_bcu: S = 2 (four DDR4 channels, two per slice)");
  initial if (TMEM_PMP != 0) $fatal(1, "otpu_fpga_top_bcu: TMEM_PMP 0 (no core_clk2x beside the gated acc_clk)");
`endif

  // ---- clocks: each DIMM's 300 MHz on one global buffer
  logic [3:0] c300_ib, c300;
  IBUFDS u_ib300_0 (.I(clk300_0_p), .IB(clk300_0_n), .O(c300_ib[0]));
  IBUFDS u_ib300_1 (.I(clk300_1_p), .IB(clk300_1_n), .O(c300_ib[1]));
  IBUFDS u_ib300_2 (.I(clk300_2_p), .IB(clk300_2_n), .O(c300_ib[2]));
  IBUFDS u_ib300_3 (.I(clk300_3_p), .IB(clk300_3_n), .O(c300_ib[3]));
  for (genvar d = 0; d < 4; d++) begin : g_bufg
    BUFG u_bufg (.I(c300_ib[d]), .O(c300[d]));
  end

  // ---- the block design
  logic        core_clk, core_rstn, xdma_aclk, xdma_aresetn, pcie_link_up;
  logic        acc_clk;                     // the accelerator's clock
  logic        acc_ce = 1'b0;               //   and its enable (core_clk; the sequencer below)
  logic [9:0]  device_temp;                 // SYSMONE4 die-temperature code (core_clk)
  logic        temp_alarm, ot_alarm;        // the SYSMON's alarms (core_clk; bd.tcl: 80 / 84 C)
  // the governor (AXI4-Lite, core_clk)
  logic [31:0] gv_awaddr, gv_araddr, gv_wdata, gv_rdata;
  logic [3:0]  gv_wstrb;
  logic [1:0]  gv_bresp, gv_rresp;
  logic gv_awvalid, gv_awready, gv_wvalid, gv_wready, gv_bvalid, gv_bready;
  logic gv_arvalid, gv_arready, gv_rvalid, gv_rready;
  // control (AXI4-Lite, core_clk)
  logic [31:0] ctl_awaddr, ctl_araddr, ctl_wdata, ctl_rdata;
  logic [3:0]  ctl_wstrb;
  logic [1:0]  ctl_bresp, ctl_rresp;
  logic ctl_awvalid, ctl_awready, ctl_wvalid, ctl_wready, ctl_bvalid, ctl_bready;
  logic ctl_arvalid, ctl_arready, ctl_rvalid, ctl_rready;
  // the BMC's UART bridge (AXI4-Lite, core_clk)
  logic [31:0] bmc_awaddr, bmc_araddr, bmc_wdata, bmc_rdata;
  logic [3:0]  bmc_wstrb;
  logic [1:0]  bmc_bresp, bmc_rresp;
  logic bmc_awvalid, bmc_awready, bmc_wvalid, bmc_wready, bmc_bvalid, bmc_bready;
  logic bmc_arvalid, bmc_arready, bmc_rvalid, bmc_rready;
  // the DIMMs' CSRs (AXI4-Lite, core_clk), [d]
  logic [3:0][31:0] mc_awaddr, mc_araddr, mc_wdata, mc_rdata;
  logic [3:0][3:0]  mc_wstrb;
  logic [3:0][1:0]  mc_bresp, mc_rresp;
  logic [3:0] mc_awvalid, mc_awready, mc_wvalid, mc_wready, mc_bvalid, mc_bready;
  logic [3:0] mc_arvalid, mc_arready, mc_rvalid, mc_rready;
  // XDMA's DMA per board channel (AXI4, 128 bits, xdma_aclk), [j]
  logic [NC-1:0][32:0]  dm_awaddr, dm_araddr;
  logic [NC-1:0][7:0]   dm_awlen, dm_arlen;
  logic [NC-1:0][127:0] dm_wdata, dm_rdata;
  logic [NC-1:0][15:0]  dm_wstrb;
  logic [NC-1:0][1:0]   dm_bresp, dm_rresp;
  logic [NC-1:0] dm_awvalid, dm_awready, dm_wvalid, dm_wready, dm_wlast, dm_bvalid, dm_bready;
  logic [NC-1:0] dm_arvalid, dm_arready, dm_rvalid, dm_rready, dm_rlast;

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
    .clk300_in(c300[1]),
    .pcie_refclk_clk_p(pcie_refclk_p), .pcie_refclk_clk_n(pcie_refclk_n),
    .pcie_perstn(pcie_perst_n),
    .pcie_mgt_rxp, .pcie_mgt_rxn, .pcie_mgt_txp, .pcie_mgt_txn,
    .pcie_link_up, .core_clk, .core_clk2x(), .core_rstn, .xdma_aclk, .xdma_aresetn, .device_temp,
    .acc_clk, .acc_clk_ce(acc_ce), .temp_alarm, .ot_alarm,
    `BD_LITE(M_AXI_CTL, ctl), `BD_LITE(M_AXI_BMC, bmc), `BD_LITE(M_AXI_GOV, gv),
    `BD_MC(0), `BD_MC(1), `BD_MC(2), `BD_MC(3),
    `BD_DMA(0), `BD_DMA(1), `BD_DMA(2), `BD_DMA(3));
`undef BD_LITE
`undef BD_MC
`undef BD_DMA

  // resets: synchronous, active high, one register each (otpu_mem_ch crosses them registered)
  logic core_rst, xrst;
  always_ff @(posedge core_clk) core_rst <= !core_rstn;
  always_ff @(posedge xdma_aclk) xrst <= !xdma_aresetn;
  // ---- the governor: SAFE after configuration, after its reset and after every trip
  logic       gov_rst = 1'b1;
  always_ff @(posedge core_clk) gov_rst <= core_rst;
  logic       acc_on;
  logic [3:0] dimm_on, dimm_ok;
  logic [1:0] uart_sel;
  logic [2:0] gov_led;
  logic signed [15:0] reg_hot;              // VCCINT's regulator (the BMC's bridge's watch)
  logic [15:0] reg_age;
  otpu_bcu_gov #(.CLK_HZ(CORE_KHZ * 1000), .HAS_REG(1'b1), .IMG_PLATFORM(PLATFORM),
                 .IMG_SLICES(S), .IMG_CARDS(1), .IMG_HASH(ARCH_HASH), .IMG_BUILD(BUILD_ID)) u_gov (
    .clk(core_clk), .rst(gov_rst), .temp(device_temp), .temp_alarm, .ot_alarm, .reg_t(reg_hot),
    .reg_age, .acc_on, .dimm_on, .dimm_ok, .uart_sel, .led(gov_led),
    .s_awaddr(gv_awaddr[11:0]), .s_awvalid(gv_awvalid), .s_awready(gv_awready),
    .s_wdata(gv_wdata), .s_wstrb(gv_wstrb), .s_wvalid(gv_wvalid), .s_wready(gv_wready),
    .s_bresp(gv_bresp), .s_bvalid(gv_bvalid), .s_bready(gv_bready),
    .s_araddr(gv_araddr[11:0]), .s_arvalid(gv_arvalid), .s_arready(gv_arready),
    .s_rdata(gv_rdata), .s_rresp(gv_rresp), .s_rvalid(gv_rvalid), .s_rready(gv_rready));

  // ---- the BMC's UART bridge: reset with its master (the SmartConnect), so neither side is ever
  // left with half an access of the other's
  otpu_bmc_uart #(.CLK_HZ(CORE_KHZ * 1000)) u_bmc (
    .clk(core_clk), .rst(core_rst),
    .s_awaddr(bmc_awaddr[15:0]), .s_awvalid(bmc_awvalid), .s_awready(bmc_awready),
    .s_wdata(bmc_wdata), .s_wstrb(bmc_wstrb), .s_wvalid(bmc_wvalid), .s_wready(bmc_wready),
    .s_bresp(bmc_bresp), .s_bvalid(bmc_bvalid), .s_bready(bmc_bready),
    .s_araddr(bmc_araddr[15:0]), .s_arvalid(bmc_arvalid), .s_arready(bmc_arready),
    .s_rdata(bmc_rdata), .s_rresp(bmc_rresp), .s_rvalid(bmc_rvalid), .s_rready(bmc_rready),
    .tx(bmc_tx), .rx(bmc_rx), .reg_hot, .reg_age);

  // ---- the DDR4 cores, [d] = DIMM C<d>
  logic [3:0]             d_sys_clk, d_sys_rst, d_ready, d_ded, d_tx;
  logic [3:0][1:0]        d_cmd_valid, d_cmd_ready, d_cmd_we, d_wdata_valid, d_wdata_ready;
  logic [3:0][1:0]        d_rdata_valid;
  logic [3:0][1:0][27:0]  d_cmd_addr;
  logic [3:0][1:0][511:0] d_wdata_data, d_rdata_data;
  logic [3:0][1:0][63:0]  d_wdata_we;
  // each core's CSR port: off (dimm_on low, or within UP_CYC of rising: otpu_axil_off answers),
  // else its sys reset in core_clk (two flip-flops), behind otpu_axil_iso. Its calibration
  // (c0_ready, sys_clk) in core_clk too, two flip-flops (the DMA gates', calib's)
  (* ASYNC_REG = "TRUE" *) logic [3:0] ld_crs1 = '1, ld_crs2 = '1;
  always_ff @(posedge core_clk) begin ld_crs1 <= d_sys_rst; ld_crs2 <= ld_crs1; end
  (* ASYNC_REG = "TRUE" *) logic [3:0] cal_c1 = '0, cal_c2 = '0;
  always_ff @(posedge core_clk) begin cal_c1 <= d_ready; cal_c2 <= cal_c1; end
  logic [3:0][$clog2(UP_CYC):0] up_cnt;
  logic [3:0] csr_off = '1, csr_rst = '1;
  logic [3:0][15:0] li_awaddr, li_araddr, lc_awaddr, lc_araddr;
  logic [3:0][31:0] li_wdata, li_rdata, lc_wdata, lc_rdata;
  logic [3:0][3:0]  li_wstrb, lc_wstrb;
  logic [3:0][1:0]  li_bresp, li_rresp, lc_bresp, lc_rresp;
  logic [3:0] li_awvalid, li_awready, li_wvalid, li_wready, li_bvalid, li_bready;
  logic [3:0] li_arvalid, li_arready, li_rvalid, li_rready;
  logic [3:0] lc_awvalid, lc_awready, lc_wvalid, lc_wready, lc_bvalid, lc_bready;
  logic [3:0] lc_arvalid, lc_arready, lc_rvalid, lc_rready;
  for (genvar d = 0; d < 4; d++) begin : g_csr
    always_ff @(posedge core_clk) begin
      if (core_rst || !dimm_on[d]) up_cnt[d] <= '0;
      else if (up_cnt[d] != ($clog2(UP_CYC) + 1)'(UP_CYC)) up_cnt[d] <= up_cnt[d] + 1'b1;
      csr_off[d] <= core_rst || !dimm_on[d] || up_cnt[d] != ($clog2(UP_CYC) + 1)'(UP_CYC);
      csr_rst[d] <= csr_off[d] || ld_crs2[d];
    end
    otpu_axil_off #(.AW(16), .ERR_RESP(2'b00)) u_off (
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
    .c0_cmd_valid(d_cmd_valid[N][0]), .c0_cmd_ready(d_cmd_ready[N][0]), \
    .c0_cmd_we(d_cmd_we[N][0]), .c0_cmd_addr(d_cmd_addr[N][0][`OTPU_BCU_AW-1:0]), \
    .c0_wdata_valid(d_wdata_valid[N][0]), .c0_wdata_ready(d_wdata_ready[N][0]), \
    .c0_wdata_data(d_wdata_data[N][0]), .c0_wdata_we(d_wdata_we[N][0]), \
    .c0_rdata_valid(d_rdata_valid[N][0]), .c0_rdata_ready(1'b1), \
    .c0_rdata_data(d_rdata_data[N][0]), .c0_ready(d_ready[N]), .c0_ded(d_ded[N]), \
    .c0b_cmd_valid(d_cmd_valid[N][1]), .c0b_cmd_ready(d_cmd_ready[N][1]), \
    .c0b_cmd_we(d_cmd_we[N][1]), .c0b_cmd_addr(d_cmd_addr[N][1][`OTPU_BCU_AW-1:0]), \
    .c0b_wdata_valid(d_wdata_valid[N][1]), .c0b_wdata_ready(d_wdata_ready[N][1]), \
    .c0b_wdata_data(d_wdata_data[N][1]), .c0b_wdata_we(d_wdata_we[N][1]), \
    .c0b_rdata_valid(d_rdata_valid[N][1]), .c0b_rdata_ready(1'b1), \
    .c0b_rdata_data(d_rdata_data[N][1]), \
    .serial_rx(uart_sel == 2'(N) ? uart_rx : 1'b1), .serial_tx(d_tx[N]), \
    .ddram_a(ddr4_c``N``_a), .ddram_we_n(ddr4_c``N``_we_n), .ddram_cas_n(ddr4_c``N``_cas_n), \
    .ddram_ras_n(ddr4_c``N``_ras_n), .ddram_act_n(ddr4_c``N``_act_n), \
    .ddram_ba(ddr4_c``N``_ba), .ddram_bg(ddr4_c``N``_bg), .ddram_cke(ddr4_c``N``_cke), \
    .ddram_odt(ddr4_c``N``_odt), .ddram_cs_n(ddr4_c``N``_cs_n), \
    .ddram_reset_n(ddr4_c``N``_reset_n), .ddram_clk_p(ddr4_c``N``_clk_p), \
    .ddram_clk_n(ddr4_c``N``_clk_n), `OTPU_BCU_LD_DM(N) .ddram_dq(ddr4_c``N``_dq), \
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

  // ---- board channel j <-> DIMM dimm(j)
  logic [NC-1:0]             calib, ded, uclk, urst;
  logic [NC-1:0][1:0]        c_cmd_valid, c_cmd_ready, c_cmd_we, c_wdata_valid, c_wdata_ready;
  logic [NC-1:0][1:0]        c_rdata_valid;
  logic [NC-1:0][1:0][24:0]  c_cmd_addr;
  logic [NC-1:0][1:0][511:0] c_wdata_data, c_rdata_data;
  logic [NC-1:0][1:0][63:0]  c_wdata_we;
  for (genvar j = 0; j < NC; j++) begin : g_map
    localparam int M = dimm(j);
    assign calib[j] = cal_c2[M] && !csr_off[M];   // STATUS CALIB0/1 (core_clk: a DIMM off is not)
    assign ded[j]   = d_ded[M];
    assign uclk[j]  = d_sys_clk[M];
    assign urst[j]  = d_sys_rst[M];
    assign d_cmd_valid[M]   = c_cmd_valid[j];
    assign c_cmd_ready[j]   = d_cmd_ready[M];
    assign d_cmd_we[M]      = c_cmd_we[j];
    // the beat in the DIMM's 16 GiB (a UDIMM's 8 GiB): the 2 GiB a slice addresses are its first
    for (genvar p = 0; p < 2; p++) begin : g_port
      assign d_cmd_addr[M][p] = {3'b000, c_cmd_addr[j][p]};
    end
    assign d_wdata_valid[M] = c_wdata_valid[j];
    assign c_wdata_ready[j] = d_wdata_ready[M];
    assign d_wdata_data[M]  = c_wdata_data[j];
    assign d_wdata_we[M]    = c_wdata_we[j];
    assign c_rdata_valid[j] = d_rdata_valid[M];
    assign c_rdata_data[j]  = d_rdata_data[M];
  end

  // ---- I2C: the board's bus (behind its mux, held out of reset) on bus 0's pins, released unless
  // I2C_CTRL drives a line low; bus 1 and ALERT0 read high (otpu_bcu_sys: CAPS bit2 clear)
  logic [3:0] i2c_lo;
  logic [1:0] i2c_lvl;
  IOBUF u_iob_scl (.IO(i2c_scl), .I(1'b0), .T(!i2c_lo[0]), .O(i2c_lvl[0]));
  IOBUF u_iob_sda (.IO(i2c_sda), .I(1'b0), .T(!i2c_lo[1]), .O(i2c_lvl[1]));
  assign i2c_mux_rst_n = 1'b1;

  // ---- the accelerator's domain: acc_clk and acc_rst. It runs only while the governor grants it
  // and every DIMM, each up (acc_want). On: the clock, then ACC_RST_CYC cycles in reset. Off: the
  // reset at once, the clock ACC_STOP_CYC cycles later, inside the governor's OFF_DLY (4096 cycles
  // between the accelerator and the DIMMs), so otpu_mem_ch's holds see its reads in flight come
  // back while the controllers run. acc_up: out of reset (its control registers answered)
  localparam int ACC_RST_CYC = 1024, ACC_STOP_CYC = 2048;
  typedef enum logic [1:0] {A_OFF, A_START, A_ON, A_STOP} acc_st_t;
  acc_st_t     acc_st = A_OFF;
  logic [11:0] acc_cnt = '0;
  logic        acc_rst = 1'b1, acc_up = 1'b0;
  wire acc_want = acc_on && &dimm_ok && !(|csr_off);
  always_ff @(posedge core_clk) begin
    acc_cnt <= acc_cnt + 1'b1;
    unique case (acc_st)
      A_OFF:   if (acc_want) begin acc_st <= A_START; acc_cnt <= '0; end
      A_START: if (!acc_want) begin acc_st <= A_STOP; acc_cnt <= '0; end
               else if (acc_cnt == 12'(ACC_RST_CYC)) acc_st <= A_ON;
      A_ON:    if (!acc_want) begin acc_st <= A_STOP; acc_cnt <= '0; end
      A_STOP:  if (acc_cnt == 12'(ACC_STOP_CYC)) acc_st <= A_OFF;
    endcase
    if (core_rst) acc_st <= A_OFF;
    acc_ce  <= acc_st != A_OFF && !core_rst;
    acc_rst <= acc_st != A_ON || core_rst;
    acc_up  <= acc_st == A_ON && !core_rst;
  end

  // the accelerator's control registers: answered here (OKAY, 0xDEAD0FF0) while it is not up
  logic [11:0] ai_awaddr, ai_araddr, ac_awaddr, ac_araddr;
  logic [31:0] ai_wdata, ai_rdata, ac_wdata, ac_rdata;
  logic [3:0]  ai_wstrb, ac_wstrb;
  logic [1:0]  ai_bresp, ai_rresp, ac_bresp, ac_rresp;
  logic ai_awvalid, ai_awready, ai_wvalid, ai_wready, ai_bvalid, ai_bready;
  logic ai_arvalid, ai_arready, ai_rvalid, ai_rready;
  logic ac_awvalid, ac_awready, ac_wvalid, ac_wready, ac_bvalid, ac_bready;
  logic ac_arvalid, ac_arready, ac_rvalid, ac_rready;
  otpu_axil_off #(.AW(12), .ERR_RESP(2'b00)) u_ctl_off (
    .clk(core_clk), .rst(core_rst), .off(!acc_up),
    .m_awaddr(ctl_awaddr[11:0]), .m_awvalid(ctl_awvalid), .m_awready(ctl_awready),
    .m_wdata(ctl_wdata), .m_wstrb(ctl_wstrb), .m_wvalid(ctl_wvalid), .m_wready(ctl_wready),
    .m_bresp(ctl_bresp), .m_bvalid(ctl_bvalid), .m_bready(ctl_bready),
    .m_araddr(ctl_araddr[11:0]), .m_arvalid(ctl_arvalid), .m_arready(ctl_arready),
    .m_rdata(ctl_rdata), .m_rresp(ctl_rresp), .m_rvalid(ctl_rvalid), .m_rready(ctl_rready),
    .s_awaddr(ai_awaddr), .s_awvalid(ai_awvalid), .s_awready(ai_awready),
    .s_wdata(ai_wdata), .s_wstrb(ai_wstrb), .s_wvalid(ai_wvalid), .s_wready(ai_wready),
    .s_bresp(ai_bresp), .s_bvalid(ai_bvalid), .s_bready(ai_bready),
    .s_araddr(ai_araddr), .s_arvalid(ai_arvalid), .s_arready(ai_arready),
    .s_rdata(ai_rdata), .s_rresp(ai_rresp), .s_rvalid(ai_rvalid), .s_rready(ai_rready));
  otpu_axil_iso #(.AW(12)) u_ctl_iso (
    .clk(core_clk), .m_rst(core_rst), .s_rst(acc_rst),
    .m_awaddr(ai_awaddr), .m_awvalid(ai_awvalid), .m_awready(ai_awready),
    .m_wdata(ai_wdata), .m_wstrb(ai_wstrb), .m_wvalid(ai_wvalid), .m_wready(ai_wready),
    .m_bresp(ai_bresp), .m_bvalid(ai_bvalid), .m_bready(ai_bready),
    .m_araddr(ai_araddr), .m_arvalid(ai_arvalid), .m_arready(ai_arready),
    .m_rdata(ai_rdata), .m_rresp(ai_rresp), .m_rvalid(ai_rvalid), .m_rready(ai_rready),
    .s_awaddr(ac_awaddr), .s_awvalid(ac_awvalid), .s_awready(ac_awready),
    .s_wdata(ac_wdata), .s_wstrb(ac_wstrb), .s_wvalid(ac_wvalid), .s_wready(ac_wready),
    .s_bresp(ac_bresp), .s_bvalid(ac_bvalid), .s_bready(ac_bready),
    .s_araddr(ac_araddr), .s_arvalid(ac_arvalid), .s_arready(ac_arready),
    .s_rdata(ac_rdata), .s_rresp(ac_rresp), .s_rvalid(ac_rvalid), .s_rready(ac_rready));

  // ---- XDMA's DMA per channel, behind otpu_axi_off: no new burst unless the channel's DIMM is
  // granted (dimm_ok), up and calibrated (off); every burst answered by the gate once the DIMM is
  // off or coming up (dead: csr_off), the channel's otpu_mem_ch XDMA side in reset meanwhile. Both
  // in core_clk, then two flip-flops into xdma_aclk. The channel's 2 GiB: sc_dma decoded bits 32:31
  logic [NC-1:0] xg_off = '1, xg_dead = '1;
  (* ASYNC_REG = "TRUE" *) logic [NC-1:0] xo1 = '1, xo2 = '1, xd1 = '1, xd2 = '1;
  logic [NC-1:0] xrst_ch = '1;
  logic [NC-1:0][30:0]  xg_awaddr, xg_araddr;
  logic [NC-1:0][7:0]   xg_awlen, xg_arlen;
  logic [NC-1:0][127:0] xg_wdata, xg_rdata;
  logic [NC-1:0][15:0]  xg_wstrb;
  logic [NC-1:0][1:0]   xg_bresp, xg_rresp;
  logic [NC-1:0] xg_awvalid, xg_awready, xg_wvalid, xg_wready, xg_wlast, xg_bvalid, xg_bready;
  logic [NC-1:0] xg_arvalid, xg_arready, xg_rvalid, xg_rready, xg_rlast;
  always_ff @(posedge xdma_aclk) begin
    xo1 <= xg_off; xo2 <= xo1; xd1 <= xg_dead; xd2 <= xd1;
    xrst_ch <= {NC{xrst}} | xd2;
  end
  for (genvar j = 0; j < NC; j++) begin : g_dma
    localparam int M = dimm(j);
    always_ff @(posedge core_clk) begin
      xg_off[j]  <= !dimm_ok[M] || csr_off[M] || !cal_c2[M];
      xg_dead[j] <= csr_off[M];
    end
    otpu_axi_off #(.AW(31), .DW(128), .NOUT(16)) u_xoff (
      .clk(xdma_aclk), .rst(xrst), .off(xo2[j]), .dead(xd2[j]),
      .m_awaddr(dm_awaddr[j][30:0]), .m_awlen(dm_awlen[j]), .m_awvalid(dm_awvalid[j]),
      .m_awready(dm_awready[j]), .m_wdata(dm_wdata[j]), .m_wstrb(dm_wstrb[j]),
      .m_wlast(dm_wlast[j]), .m_wvalid(dm_wvalid[j]), .m_wready(dm_wready[j]),
      .m_bresp(dm_bresp[j]), .m_bvalid(dm_bvalid[j]), .m_bready(dm_bready[j]),
      .m_araddr(dm_araddr[j][30:0]), .m_arlen(dm_arlen[j]), .m_arvalid(dm_arvalid[j]),
      .m_arready(dm_arready[j]), .m_rdata(dm_rdata[j]), .m_rresp(dm_rresp[j]),
      .m_rlast(dm_rlast[j]), .m_rvalid(dm_rvalid[j]), .m_rready(dm_rready[j]),
      .s_awaddr(xg_awaddr[j]), .s_awlen(xg_awlen[j]), .s_awvalid(xg_awvalid[j]),
      .s_awready(xg_awready[j]), .s_wdata(xg_wdata[j]), .s_wstrb(xg_wstrb[j]),
      .s_wlast(xg_wlast[j]), .s_wvalid(xg_wvalid[j]), .s_wready(xg_wready[j]),
      .s_bresp(xg_bresp[j]), .s_bvalid(xg_bvalid[j]), .s_bready(xg_bready[j]),
      .s_araddr(xg_araddr[j]), .s_arlen(xg_arlen[j]), .s_arvalid(xg_arvalid[j]),
      .s_arready(xg_arready[j]), .s_rdata(xg_rdata[j]), .s_rresp(xg_rresp[j]),
      .s_rlast(xg_rlast[j]), .s_rvalid(xg_rvalid[j]), .s_rready(xg_rready[j]));
  end

  // ---- the accelerator and XDMA on the channels
  logic [2:0] board_led;
  otpu_bcu_sys #(.S(S), .D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .ACT_BLOCKS(ACT_BLOCKS),
                 .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
                 .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
                 .MXU_CL(MXU_CL), .VPU_CL(VPU_CL), .ULANES(ULANES), .TRACE_DEPTH(TRACE_DEPTH),
                 .PQ_WIN(PQ_WIN), .CHASH(CHASH), .DSTEP(DSTEP), .CORE_KHZ(CORE_KHZ),
                 .BUILD_ID(BUILD_ID), .SEQ_WC(SEQ_WC), .DDR_MTS(DDR_MTS), .PLATFORM(PLATFORM),
                 .ARCH_HASH(ARCH_HASH), .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ),
                 .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH), .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP),
                 .TMEM_GST(TMEM_GST), .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM)) u_sys (
    .clk(acc_clk), .clk_f(1'b0), .rst(acc_rst), .ctl_mrst(core_rst), .xclk(xdma_aclk), .xrst(xrst_ch),
    .calib, .ded, .temp({device_temp, 2'b00}), .led(board_led), .i2c_lo,
    .i2c_pin({3'b111, i2c_lvl}),
    .s_ctl_awaddr(ac_awaddr), .s_ctl_awvalid(ac_awvalid), .s_ctl_awready(ac_awready),
    .s_ctl_wdata(ac_wdata), .s_ctl_wstrb(ac_wstrb), .s_ctl_wvalid(ac_wvalid),
    .s_ctl_wready(ac_wready), .s_ctl_bresp(ac_bresp), .s_ctl_bvalid(ac_bvalid),
    .s_ctl_bready(ac_bready), .s_ctl_araddr(ac_araddr), .s_ctl_arvalid(ac_arvalid),
    .s_ctl_arready(ac_arready), .s_ctl_rdata(ac_rdata), .s_ctl_rresp(ac_rresp),
    .s_ctl_rvalid(ac_rvalid), .s_ctl_rready(ac_rready),
    .x_awvalid(xg_awvalid), .x_awready(xg_awready), .x_awaddr(xg_awaddr), .x_awlen(xg_awlen),
    .x_wvalid(xg_wvalid), .x_wready(xg_wready), .x_wdata(xg_wdata), .x_wstrb(xg_wstrb),
    .x_wlast(xg_wlast), .x_bvalid(xg_bvalid), .x_bready(xg_bready), .x_bresp(xg_bresp),
    .x_arvalid(xg_arvalid), .x_arready(xg_arready), .x_araddr(xg_araddr), .x_arlen(xg_arlen),
    .x_rvalid(xg_rvalid), .x_rready(xg_rready), .x_rdata(xg_rdata), .x_rresp(xg_rresp),
    .x_rlast(xg_rlast),
    .uclk, .urst,
    .c_cmd_valid, .c_cmd_ready, .c_cmd_we, .c_cmd_addr, .c_wdata_valid, .c_wdata_ready,
    .c_wdata_data, .c_wdata_we, .c_rdata_valid, .c_rdata_data);

  // LEDs: led[0] the governor alive (1 Hz), led[1] PCIe link up and the governor in RUN, led[2]
  // the governor's WARN or a thermal trip, or the accelerator runs or halted cleanly
  assign led = {gov_led[2] | board_led[2] | board_led[1], pcie_link_up & gov_led[1], gov_led[0]};
endmodule
