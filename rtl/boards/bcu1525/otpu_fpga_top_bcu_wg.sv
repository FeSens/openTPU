// FPGA top for the BCU1525 at WG 8 (xcvu9p-fsgd2104-2L-e; targets/bcu1525-wg.toml,
// docs/bcu1525_wg.md): the block design otpu_bdw (XDMA Gen3 x16, the clocks, the SYSMON, the control
// interconnect, XDMA's DMA split by DIMM, and the four DIMMs' DDR4-2400 controllers, each behind an
// AXI Clock Converter from aclk: boards/bcu1525/vivado/bd_wg.tcl), the governor (otpu_bcu_gov, BAR0
// 0x10000), the BMC's UART bridge (otpu_bmc_uart, 0x20000), and otpu_bcu_wg_sys: the accelerator
// (otpu_board, one slice of 8 channels at WG 8, the WG targets') through its bridges and the DIMMs' pairs,
// each pair with the host's DMA through a DIMM's otpu_mig_hmux onto its port (S_AXI_P<d>, aclk; no
// SmartConnect on the accelerator's stream). The parameters other than the board's own are the
// target's, otpu_board's OTPU_BOARD and the bridges' OTPU_* defines (otpu_bcu_wg_defines.vh:
// boards/bcu1525/scripts/make_wg.py from the target's rtl_params, read before this file).
//
// Channels: virtual channel j = 2d + s (the slice's channel j, CB 128) is DIMM C<d>'s half s, a
// channel of 2^CL bytes (CL = CAW + 7: CAW 22 512 MiB, 25 4 GiB): its byte o at the DIMM's
// {o[CL-1:12], s, o[11:0]} (otpu_mig_pair). The host's DMA reaches channel j at card address
// j x 2^CL (the WG targets' layout at CAW 22: j x 512 MiB; at 25 j x 4 GiB): sc_dma decodes DIMM d's
// 2^(CL+1) bytes at d x 2^(CL+1) (M_AXI_DMA<d>, 64-bit addresses: 1 GiB windows at CAW 22, 8 GiB
// at 25), and here each burst's offset in its window goes through the same permutation (bit CL
// the channel; a burst never crosses 4 KiB, so it stays whole) into S_AXI_H<d>, and out of the
// block design at aclk (sc_h<d>, M_AXI_H<d>) into DIMM d's hmux beside its pair. CAW is
// OTPU_WG_CAW, make_wg.py's from the target's channel_bytes (22 without it): the MIGs' AXI side is
// 33 bits (an 8 GB DIMM), CAW 26 (16 GB DIMMs) would need it at 34 (bd_wg.tcl). BHW, the pairs'
// bank-half window (otpu_mig_pair; OTPU_WG_BHW, make_wg.py's from the target's BHW, 0 without
// it): the host's bytes there take the same swap of the MIG's bank-high bit and the window's top.
//
// S 2 (OTPU_WG_S, targets/bcu1525-s2.toml; docs/bcu1525_s2.md): two slices, each on
// two DIMMs, four channels a DIMM (CL = CAW + 7, CAW 24: 2 GiB a channel). Board channel g (slice
// g / 8's channel g mod 8) is DIMM slot g / 4's quarter g mod 4, slot i on DIMM DMAP[2 i +: 2]
// (OTPU_WG_DMAP, make_wg.py's floorplan: boards/bcu1525/scripts/bcu_fp.py). The host's DMA reaches
// board channel g at g x 2^CL: sc_dma's window i (2^DL bytes at i x 2^DL, DL = CL + log2(2 S), the
// DIMM's byte address bits: 8 GiB at either S) is slot i's, its burst's channel bits a[DL-1:CL]
// moved to bit 12 (remap), and it goes to DIMM DMAP[2 i +: 2]'s path (g_dma: slot_of). The hmuxes
// take the host's ID at bit 0 (HB), the pairs' four channels the top two bits.
//
// Clocks (bd_wg.tcl): core_clk (CORE_KHZ: the governor, the gates, the control side), acc_clk and
// acc_clk2x (core_clk's and twice it, behind the sequencer's enable: the accelerator's, TMEM_PMP
// 1; both from the MMCM's 2x output through a BUFGCE_DIV pair here, /2 and /1 on one enable, and
// the enable's register), aclk (300 MHz, ungated: the bridges' AXI side, the pairs, the hmuxes),
// xdma_aclk (250 MHz), each MIG's ui_clk (inside the block design: the clock converters cross
// into it). CORE_KHZ1 (create_project_wg.tcl: 133333 at 150 MHz) is the same netlist's fallback
// clock: boards/bcu1525/vivado/clk133.tcl (OTPU_CLK_HOOK) sets the MMCM's 2x and core outputs to
// 4.5 and 9 of its 1200 MHz VCO, and the INIT of the select flip-flops (otpu_ctrl.sv's
// otpu_clk_k) of the governor's and the watch's millisecond, the UART's bit and CORE_KHZ.
// Power (as otpu_fpga_top_bcu, docs/bcu1525.md section 14): after configuration everything is off
// (the governor's SAFE): the accelerator's clocks stopped and its domain in reset, every MIG in
// reset (mig_rst: its MMCM stopped, its DIMM unclocked; it calibrates again when turned on). A
// DIMM is live UP_CYC core cycles after the governor turns it on (dimm_live: its AXI path, hmux and
// its bridges' AXI side out of reset), up once its MIG has calibrated too (dimm_up: its DMA gate
// open, STATUS CALIB). The accelerator runs only while the governor grants it and every DIMM, each
// up; its sequencer (on: the clocks, then ACC_RST_CYC cycles of reset; off: the reset at once, the
// clocks ACC_STOP_CYC cycles later, inside the governor's OFF_DLY) lets the bridges' reset drain
// what they have in flight while the DIMMs still run. Its control registers are behind
// otpu_axil_off (OKAY, 0xDEAD0FF0 while it is not up) and otpu_axil_iso; each DIMM's DMA behind
// otpu_axi_off (no new burst unless the DIMM is granted and up; every burst answered by the gate
// once it is not live).
// Resets: core_rst from the block design (MMCM lock, PCIe PERST#), xrst from XDMA's axi_aresetn.
// The DIMM pins the MIG does not drive (a single-rank UDIMM): par 0, the other ranks' chip selects
// high, cke1 / odt1 low. The board's UART (uart_tx) idles (no controller BIOS).
`ifndef OTPU_WG_CAW                // the channels' beat address bits (make_wg.py)
`define OTPU_WG_CAW 22
`endif
`ifndef OTPU_WG_S                  // the slices (make_wg.py: the target's S)
`define OTPU_WG_S 1
`endif
`ifndef OTPU_WG_DMAP               // the DIMM slots' DIMMs (make_wg.py --fp; S 1: the identity)
`define OTPU_WG_DMAP 8'hE4
`endif
`ifndef OTPU_WG_PLATFORM           // the image's identity for the governor (make_wg.py: the board's
`define OTPU_WG_PLATFORM 0         // PLATFORM, ARCH_HASH and BUILD_ID, as OTPU_BOARD gives otpu_board)
`define OTPU_WG_CFG_HASH 32'h0
`define OTPU_WG_BUILD_ID 32'h0
`endif
`ifndef OTPU_WG_BHW                // the pairs' bank-half window, a channel's bytes' log2 (make_wg.py)
`define OTPU_WG_BHW 0
`endif
module otpu_fpga_top_bcu_wg #(
  parameter int CORE_KHZ   = 100000,        // core_clk as the block design makes it
  parameter int CORE_KHZ1  = CORE_KHZ,      // its fallback (clk133.tcl), or CORE_KHZ: none
  parameter int PCIE_LANES = 16,            // Gen3 x16 (or x8: bd_wg.tcl)
  parameter int UP_CYC     = 1 << 17,       // a DIMM live this long after turning on
  parameter int S          = `OTPU_WG_S,    // slices (otpu_bcu_wg_sys S)
  parameter logic [32*S-1:0] RLY = '0,      // channel j's bridge pipeline N (make_wg.py --slr / --fp)
  parameter logic [7:0] DMAP = `OTPU_WG_DMAP, // DIMM slot i's DIMM (otpu_bcu_wg_sys DMAP)
  parameter int CAW        = `OTPU_WG_CAW,  // the channels' beat address bits (otpu_bcu_wg_sys)
  parameter int BHW        = `OTPU_WG_BHW,  // the pairs' bank-half window (otpu_mig_pair; 0: none)
  parameter int BHB        = 16,            // and the MIG map's bank-high bit (WG_MIGMAP b15's)
  parameter int ACC_RA     = S == 2 ? 2 : 0 // acc_rst's last stages in acc_clk (0: all core_clk's)
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
  output logic        uart_tx,
  output logic        bmc_tx,               // the board management controller's UART (BB19 / BA19)
  input  logic        bmc_rx,
`ifdef OTPU_WG_X2
  // x2: the same S=2 card plus the two QSFP28 links. U9 alone supplies both quads' reference;
  // no SI5335 control pins, U12 buffer, or second STARTUPE3 owner (boards/bcu1525/x2/README.md).
  input  wire         qsfp0_mgt_refclk_1_p,
  input  wire         qsfp0_mgt_refclk_1_n,
  input  wire [3:0]   qsfp0_rx_p, qsfp0_rx_n, qsfp1_rx_p, qsfp1_rx_n,
  output wire [3:0]   qsfp0_tx_p, qsfp0_tx_n, qsfp1_tx_p, qsfp1_tx_n,
  output wire        qsfp0_resetl, qsfp0_lpmode, qsfp0_modsell,
  output wire        qsfp1_resetl, qsfp1_lpmode, qsfp1_modsell,
`endif
  // ---- PCIe Gen3 (GTY quads 224-227: XDMA's select_quad)
  input  logic        pcie_refclk_p,
  input  logic        pcie_refclk_n,
  input  logic        pcie_perst_n,
  input  logic [PCIE_LANES-1:0] pcie_mgt_rxp,
  input  logic [PCIE_LANES-1:0] pcie_mgt_rxn,
  output logic [PCIE_LANES-1:0] pcie_mgt_txp,
  output logic [PCIE_LANES-1:0] pcie_mgt_txn,
  // ---- the four DDR4 DIMMs (C0 .. C3: the MIGs' pins, constraints/bcu1525_wg_pins.xdc; par,
  // cke1, odt1, cs_hi: the pins a single-rank UDIMM's controller does not drive)
`define OTPU_WG_DIMM(N) \
  output logic [16:0] ddr4_c``N``_adr, \
  output logic [1:0]  ddr4_c``N``_ba, \
  output logic [1:0]  ddr4_c``N``_bg, \
  output logic        ddr4_c``N``_act_n, \
  output logic        ddr4_c``N``_ck_t, \
  output logic        ddr4_c``N``_ck_c, \
  output logic        ddr4_c``N``_cke, \
  output logic        ddr4_c``N``_cs_n, \
  output logic        ddr4_c``N``_odt, \
  output logic        ddr4_c``N``_reset_n, \
  inout  wire  [7:0]  ddr4_c``N``_dm_n, \
  inout  wire  [63:0] ddr4_c``N``_dq, \
  inout  wire  [7:0]  ddr4_c``N``_dqs_t, \
  inout  wire  [7:0]  ddr4_c``N``_dqs_c, \
  output logic        ddr4_c``N``_par, \
  output logic        ddr4_c``N``_cke1, \
  output logic        ddr4_c``N``_odt1, \
  output logic [2:0]  ddr4_c``N``_cs_hi
  `OTPU_WG_DIMM(0),
  `OTPU_WG_DIMM(1),
  `OTPU_WG_DIMM(2),
  `OTPU_WG_DIMM(3)
`undef OTPU_WG_DIMM
);
  localparam int AXB = PCIE_LANES == 16 ? 512 : 256;   // XDMA's DMA width

  // ---- the block design
  logic        core_clk, core_rstn, xdma_aclk, xdma_aresetn, pcie_link_up;
  logic        acc_clk, acc_clk2x, aclk;
`ifdef OTPU_WG_X2
  wire         init_clk;                 // ungated 100 MHz, Aurora's reset / DRP sequencing
`endif
  logic        acc_clk2x_u;                 // acc_clk2x before its buffers (the MMCM's output)
  logic        acc_ce = 1'b0;               // the accelerator's clocks' enable (core_clk)
  logic [9:0]  device_temp;
  logic        temp_alarm, ot_alarm;
  logic [3:0]  mig_rst = '1, sc_rstn, calib;
  // AXI4-Lite masters (core_clk)
  logic [31:0] gv_awaddr, gv_araddr, gv_wdata, gv_rdata;
  logic [3:0]  gv_wstrb;
  logic [1:0]  gv_bresp, gv_rresp;
  logic gv_awvalid, gv_awready, gv_wvalid, gv_wready, gv_bvalid, gv_bready;
  logic gv_arvalid, gv_arready, gv_rvalid, gv_rready;
  logic [31:0] ctl_awaddr, ctl_araddr, ctl_wdata, ctl_rdata;
  logic [3:0]  ctl_wstrb;
  logic [1:0]  ctl_bresp, ctl_rresp;
  logic ctl_awvalid, ctl_awready, ctl_wvalid, ctl_wready, ctl_bvalid, ctl_bready;
  logic ctl_arvalid, ctl_arready, ctl_rvalid, ctl_rready;
  logic [31:0] bmc_awaddr, bmc_araddr, bmc_wdata, bmc_rdata;
  logic [3:0]  bmc_wstrb;
  logic [1:0]  bmc_bresp, bmc_rresp;
  logic bmc_awvalid, bmc_awready, bmc_wvalid, bmc_wready, bmc_bvalid, bmc_bready;
  logic bmc_arvalid, bmc_arready, bmc_rvalid, bmc_rready;
  localparam int CL  = CAW + 7;             // a channel's byte address bits
  localparam int DL  = CL + $clog2(2 * S);  // a DIMM's (its 2 S channels'): a DMA window's
  localparam int MAW = DL >= 33 ? DL : 33;  // S_AXI_H / S_AXI_P (the MIGs' AXI side)
  // DIMM d's slot (the DMA window whose channels it holds: DMAP's inverse)
  function automatic int slot_of(input int d);
    for (int i = 0; i < 4; i++) if (int'(DMAP[2 * i +: 2]) == d) return i;
    return 0;
  endfunction
  // XDMA's DMA per DIMM (dm_*: from sc_dma; hg_*: gated and remapped, into sc_mem<d>), xdma_aclk
  logic [3:0][63:0]    dm_awaddr, dm_araddr;
  logic [3:0][7:0]     dm_awlen, dm_arlen, hg_awlen, hg_arlen;
  logic [3:0][AXB-1:0] dm_wdata, dm_rdata, hg_wdata, hg_rdata;
  logic [3:0][AXB/8-1:0] dm_wstrb, hg_wstrb;
  logic [3:0][1:0]     dm_bresp, dm_rresp, hg_bresp, hg_rresp;
  logic [3:0] dm_awvalid, dm_awready, dm_wvalid, dm_wready, dm_wlast, dm_bvalid, dm_bready;
  logic [3:0] dm_arvalid, dm_arready, dm_rvalid, dm_rready, dm_rlast;
  logic [3:0][DL-1:0]  hg_awaddr, hg_araddr;
  logic [3:0] hg_awvalid, hg_awready, hg_wvalid, hg_wready, hg_wlast, hg_bvalid, hg_bready;
  logic [3:0] hg_arvalid, hg_arready, hg_rvalid, hg_rready, hg_rlast;
  // the pairs' ports (otpu_bcu_wg_sys m_*), aclk
  logic [3:0]        p_awvalid, p_awready, p_wvalid, p_wready, p_wlast, p_bvalid, p_bready;
  logic [3:0]        p_arvalid, p_arready, p_rvalid, p_rready, p_rlast;
  logic [3:0][3:0]   p_awid, p_arid, p_bid, p_rid;
  logic [3:0][DL-1:0] p_awaddr, p_araddr;
  logic [3:0][7:0]   p_awlen, p_arlen;
  logic [3:0][511:0] p_wdata, p_rdata;
  logic [3:0][63:0]  p_wstrb;
  logic [3:0][1:0]   p_bresp, p_rresp;
  // the host's DMA at aclk (M_AXI_H<d>: sc_h<d>'s, 512 bits) into the hmuxes
  logic [3:0]          ha_awvalid, ha_awready, ha_wvalid, ha_wready, ha_wlast, ha_bvalid, ha_bready;
  logic [3:0]          ha_arvalid, ha_arready, ha_rvalid, ha_rready, ha_rlast;
  logic [3:0][MAW-1:0] ha_awaddr, ha_araddr;
  logic [3:0][7:0]     ha_awlen, ha_arlen;
  logic [3:0][511:0]   ha_wdata, ha_rdata;
  logic [3:0][63:0]    ha_wstrb;
  logic [3:0][1:0]     ha_bresp, ha_rresp;
  // the DIMMs' ports (the hmuxes' m_*: S_AXI_P<d>), aclk
  logic [3:0]          sp_awvalid, sp_awready, sp_wvalid, sp_wready, sp_wlast, sp_bvalid, sp_bready;
  logic [3:0]          sp_arvalid, sp_arready, sp_rvalid, sp_rready, sp_rlast;
  logic [3:0][3:0]     sp_awid, sp_arid, sp_bid, sp_rid;
  logic [3:0][MAW-1:0] sp_awaddr, sp_araddr;
  logic [3:0][7:0]     sp_awlen, sp_arlen;
  logic [3:0][511:0]   sp_wdata, sp_rdata;
  logic [3:0][63:0]    sp_wstrb;
  logic [3:0][1:0]     sp_bresp, sp_rresp;

  // the DIMM's byte of offset a in its window of 2^DL: virtual channel s = a[DL-1:CL], its byte
  // a[CL-1:0], through the bank-half window (otpu_mig_pair's dimm and bhw: its bits above 12, so a
  // burst stays whole; the channel byte's bit k from 12 up at the DIMM's k + DL - CL)
  localparam int WB = BHW != 0 ? BHW : 14, HB = BHW != 0 ? BHB : 13;    // (BHW 0: indices in range)
  function automatic logic [MAW-1:0] remap(input logic [DL-1:0] a);
    logic [DL-1:0] d;
    d = {a[CL-1:12], a[DL-1:CL], a[11:0]};
    if (BHW != 0 && a[CL-1:WB] == '0) begin
      d[HB] = a[WB-1];
      d[WB-1+DL-CL] = a[HB-(DL-CL)];
    end
    return MAW'(d);
  endfunction

`define BD_LITE(P, S) \
    .P``_awaddr(S``_awaddr), .P``_awvalid(S``_awvalid), .P``_awready(S``_awready), \
    .P``_wdata(S``_wdata), .P``_wstrb(S``_wstrb), .P``_wvalid(S``_wvalid), \
    .P``_wready(S``_wready), .P``_bresp(S``_bresp), .P``_bvalid(S``_bvalid), \
    .P``_bready(S``_bready), .P``_araddr(S``_araddr), .P``_arvalid(S``_arvalid), \
    .P``_arready(S``_arready), .P``_rdata(S``_rdata), .P``_rresp(S``_rresp), \
    .P``_rvalid(S``_rvalid), .P``_rready(S``_rready)
`define BD_DIMM(D) \
    .c300_``D``_clk_p(clk300_``D``_p), .c300_``D``_clk_n(clk300_``D``_n), \
    .ddr4_c``D``_adr(ddr4_c``D``_adr), .ddr4_c``D``_ba(ddr4_c``D``_ba), \
    .ddr4_c``D``_bg(ddr4_c``D``_bg), .ddr4_c``D``_act_n(ddr4_c``D``_act_n), \
    .ddr4_c``D``_ck_t(ddr4_c``D``_ck_t), .ddr4_c``D``_ck_c(ddr4_c``D``_ck_c), \
    .ddr4_c``D``_cke(ddr4_c``D``_cke), .ddr4_c``D``_cs_n(ddr4_c``D``_cs_n), \
    .ddr4_c``D``_odt(ddr4_c``D``_odt), .ddr4_c``D``_reset_n(ddr4_c``D``_reset_n), \
    .ddr4_c``D``_dm_n(ddr4_c``D``_dm_n), .ddr4_c``D``_dq(ddr4_c``D``_dq), \
    .ddr4_c``D``_dqs_t(ddr4_c``D``_dqs_t), .ddr4_c``D``_dqs_c(ddr4_c``D``_dqs_c), \
    .mig_rst``D(mig_rst[D]), .sc_rstn``D(sc_rstn[D]), .calib``D(calib[D]), \
    .M_AXI_DMA``D``_awaddr(dm_awaddr[D]), .M_AXI_DMA``D``_awlen(dm_awlen[D]), \
    .M_AXI_DMA``D``_awvalid(dm_awvalid[D]), .M_AXI_DMA``D``_awready(dm_awready[D]), \
    .M_AXI_DMA``D``_wdata(dm_wdata[D]), .M_AXI_DMA``D``_wstrb(dm_wstrb[D]), \
    .M_AXI_DMA``D``_wlast(dm_wlast[D]), .M_AXI_DMA``D``_wvalid(dm_wvalid[D]), \
    .M_AXI_DMA``D``_wready(dm_wready[D]), .M_AXI_DMA``D``_bresp(dm_bresp[D]), \
    .M_AXI_DMA``D``_bvalid(dm_bvalid[D]), .M_AXI_DMA``D``_bready(dm_bready[D]), \
    .M_AXI_DMA``D``_araddr(dm_araddr[D]), .M_AXI_DMA``D``_arlen(dm_arlen[D]), \
    .M_AXI_DMA``D``_arvalid(dm_arvalid[D]), .M_AXI_DMA``D``_arready(dm_arready[D]), \
    .M_AXI_DMA``D``_rdata(dm_rdata[D]), .M_AXI_DMA``D``_rresp(dm_rresp[D]), \
    .M_AXI_DMA``D``_rlast(dm_rlast[D]), .M_AXI_DMA``D``_rvalid(dm_rvalid[D]), \
    .M_AXI_DMA``D``_rready(dm_rready[D]), \
    .S_AXI_H``D``_awaddr(remap(hg_awaddr[D])), .S_AXI_H``D``_awlen(hg_awlen[D]), \
    .S_AXI_H``D``_awsize(3'($clog2(AXB / 8))), .S_AXI_H``D``_awburst(2'b01), \
    .S_AXI_H``D``_awvalid(hg_awvalid[D]), .S_AXI_H``D``_awready(hg_awready[D]), \
    .S_AXI_H``D``_wdata(hg_wdata[D]), .S_AXI_H``D``_wstrb(hg_wstrb[D]), \
    .S_AXI_H``D``_wlast(hg_wlast[D]), .S_AXI_H``D``_wvalid(hg_wvalid[D]), \
    .S_AXI_H``D``_wready(hg_wready[D]), .S_AXI_H``D``_bresp(hg_bresp[D]), \
    .S_AXI_H``D``_bvalid(hg_bvalid[D]), .S_AXI_H``D``_bready(hg_bready[D]), \
    .S_AXI_H``D``_araddr(remap(hg_araddr[D])), .S_AXI_H``D``_arlen(hg_arlen[D]), \
    .S_AXI_H``D``_arsize(3'($clog2(AXB / 8))), .S_AXI_H``D``_arburst(2'b01), \
    .S_AXI_H``D``_arvalid(hg_arvalid[D]), .S_AXI_H``D``_arready(hg_arready[D]), \
    .S_AXI_H``D``_rdata(hg_rdata[D]), .S_AXI_H``D``_rresp(hg_rresp[D]), \
    .S_AXI_H``D``_rlast(hg_rlast[D]), .S_AXI_H``D``_rvalid(hg_rvalid[D]), \
    .S_AXI_H``D``_rready(hg_rready[D]), \
    .M_AXI_H``D``_awaddr(ha_awaddr[D]), .M_AXI_H``D``_awlen(ha_awlen[D]), \
    .M_AXI_H``D``_awvalid(ha_awvalid[D]), .M_AXI_H``D``_awready(ha_awready[D]), \
    .M_AXI_H``D``_wdata(ha_wdata[D]), .M_AXI_H``D``_wstrb(ha_wstrb[D]), \
    .M_AXI_H``D``_wlast(ha_wlast[D]), .M_AXI_H``D``_wvalid(ha_wvalid[D]), \
    .M_AXI_H``D``_wready(ha_wready[D]), .M_AXI_H``D``_bresp(ha_bresp[D]), \
    .M_AXI_H``D``_bvalid(ha_bvalid[D]), .M_AXI_H``D``_bready(ha_bready[D]), \
    .M_AXI_H``D``_araddr(ha_araddr[D]), .M_AXI_H``D``_arlen(ha_arlen[D]), \
    .M_AXI_H``D``_arvalid(ha_arvalid[D]), .M_AXI_H``D``_arready(ha_arready[D]), \
    .M_AXI_H``D``_rdata(ha_rdata[D]), .M_AXI_H``D``_rresp(ha_rresp[D]), \
    .M_AXI_H``D``_rlast(ha_rlast[D]), .M_AXI_H``D``_rvalid(ha_rvalid[D]), \
    .M_AXI_H``D``_rready(ha_rready[D]), \
    .S_AXI_P``D``_awid(sp_awid[D]), .S_AXI_P``D``_awaddr(sp_awaddr[D]), \
    .S_AXI_P``D``_awlen(sp_awlen[D]), .S_AXI_P``D``_awsize(3'd6), .S_AXI_P``D``_awburst(2'b01), \
    .S_AXI_P``D``_awvalid(sp_awvalid[D]), .S_AXI_P``D``_awready(sp_awready[D]), \
    .S_AXI_P``D``_wdata(sp_wdata[D]), .S_AXI_P``D``_wstrb(sp_wstrb[D]), \
    .S_AXI_P``D``_wlast(sp_wlast[D]), .S_AXI_P``D``_wvalid(sp_wvalid[D]), \
    .S_AXI_P``D``_wready(sp_wready[D]), .S_AXI_P``D``_bid(sp_bid[D]), \
    .S_AXI_P``D``_bresp(sp_bresp[D]), .S_AXI_P``D``_bvalid(sp_bvalid[D]), \
    .S_AXI_P``D``_bready(sp_bready[D]), \
    .S_AXI_P``D``_arid(sp_arid[D]), .S_AXI_P``D``_araddr(sp_araddr[D]), \
    .S_AXI_P``D``_arlen(sp_arlen[D]), .S_AXI_P``D``_arsize(3'd6), .S_AXI_P``D``_arburst(2'b01), \
    .S_AXI_P``D``_arvalid(sp_arvalid[D]), .S_AXI_P``D``_arready(sp_arready[D]), \
    .S_AXI_P``D``_rid(sp_rid[D]), .S_AXI_P``D``_rdata(sp_rdata[D]), \
    .S_AXI_P``D``_rresp(sp_rresp[D]), .S_AXI_P``D``_rlast(sp_rlast[D]), \
    .S_AXI_P``D``_rvalid(sp_rvalid[D]), .S_AXI_P``D``_rready(sp_rready[D])
  otpu_bdw_wrapper u_bd (
    .pcie_refclk_clk_p(pcie_refclk_p), .pcie_refclk_clk_n(pcie_refclk_n),
    .pcie_perstn(pcie_perst_n),
    .pcie_mgt_rxp, .pcie_mgt_rxn, .pcie_mgt_txp, .pcie_mgt_txn,
    .pcie_link_up, .core_clk, .core_rstn, .acc_clk2x(acc_clk2x_u),
    .aclk, .xdma_aclk, .xdma_aresetn, .device_temp, .temp_alarm, .ot_alarm,
`ifdef OTPU_WG_X2
    .init_clk,
`endif
    `BD_LITE(M_AXI_CTL, ctl), `BD_LITE(M_AXI_BMC, bmc), `BD_LITE(M_AXI_GOV, gv),
    `BD_DIMM(0), `BD_DIMM(1), `BD_DIMM(2), `BD_DIMM(3));
`undef BD_LITE
`undef BD_DIMM

  // resets: synchronous, active high, one register each
  logic core_rst, xrst;
  always_ff @(posedge core_clk) core_rst <= !core_rstn;
  always_ff @(posedge xdma_aclk) xrst <= !xdma_aresetn;
  // ---- the governor: SAFE after configuration, after its reset and after every trip
  logic       gov_rst = 1'b1;
  always_ff @(posedge core_clk) gov_rst <= core_rst;
  logic       acc_on;
  logic [3:0] dimm_on, dimm_ok;
  logic [2:0] gov_led;
  logic signed [15:0] reg_hot;
  logic [15:0] reg_age;
`ifdef OTPU_WG_X2
  localparam int GOV_CARDS = 2;
`else
  localparam int GOV_CARDS = 1;
`endif
  otpu_bcu_gov #(.CLK_HZ(CORE_KHZ * 1000), .CLK_HZ1(CORE_KHZ1 * 1000), .HAS_REG(1'b1),
                 .IMG_PLATFORM(`OTPU_WG_PLATFORM), .IMG_SLICES(S), .IMG_CARDS(GOV_CARDS),
                 .IMG_HASH(`OTPU_WG_CFG_HASH), .IMG_BUILD(`OTPU_WG_BUILD_ID)) u_gov (
    .clk(core_clk), .rst(gov_rst), .temp(device_temp), .temp_alarm, .ot_alarm, .reg_t(reg_hot),
    .reg_age, .acc_on, .dimm_on, .dimm_ok, .uart_sel(), .led(gov_led),
    .s_awaddr(gv_awaddr[11:0]), .s_awvalid(gv_awvalid), .s_awready(gv_awready),
    .s_wdata(gv_wdata), .s_wstrb(gv_wstrb), .s_wvalid(gv_wvalid), .s_wready(gv_wready),
    .s_bresp(gv_bresp), .s_bvalid(gv_bvalid), .s_bready(gv_bready),
    .s_araddr(gv_araddr[11:0]), .s_arvalid(gv_arvalid), .s_arready(gv_arready),
    .s_rdata(gv_rdata), .s_rresp(gv_rresp), .s_rvalid(gv_rvalid), .s_rready(gv_rready));

  // ---- the BMC's UART bridge, reset with its master (the SmartConnect)
  otpu_bmc_uart #(.CLK_HZ(CORE_KHZ * 1000), .CLK_HZ1(CORE_KHZ1 * 1000)) u_bmc (
    .clk(core_clk), .rst(core_rst),
    .s_awaddr(bmc_awaddr[15:0]), .s_awvalid(bmc_awvalid), .s_awready(bmc_awready),
    .s_wdata(bmc_wdata), .s_wstrb(bmc_wstrb), .s_wvalid(bmc_wvalid), .s_wready(bmc_wready),
    .s_bresp(bmc_bresp), .s_bvalid(bmc_bvalid), .s_bready(bmc_bready),
    .s_araddr(bmc_araddr[15:0]), .s_arvalid(bmc_arvalid), .s_arready(bmc_arready),
    .s_rdata(bmc_rdata), .s_rresp(bmc_rresp), .s_rvalid(bmc_rvalid), .s_rready(bmc_rready),
    .tx(bmc_tx), .rx(bmc_rx), .reg_hot, .reg_age);

  // ---- DCI calibrated again after configuration. In the multifunction banks (65: C1's address
  // and command and one data byte, 70: C3's) DCI needs a DCIRESET (DRC DCIRST-1). The MIG adds its
  // own when it knows its banks need one, but generated without their pins these did not. One
  // DCIRESET in each SLR's configuration site (a VU9P takes three), in reset for DCI_RST_CYC core
  // cycles after core_rst. Every MIG stays in reset until all three are LOCKED (two flip-flops
  // into core_clk), or DCI_WAIT core cycles have passed without that (a DCIRESET never locking:
  // the DIMMs as without one).
  localparam int DCI_RST_CYC = 256, DCI_WAIT = 1 << 20;
  logic [$clog2(DCI_RST_CYC):0] dci_cnt = '0;
  logic [$clog2(DCI_WAIT):0]    dci_wt = '0;
  logic       dci_rst = 1'b1, dci_ok = 1'b0;
  logic [2:0] dci_lk;
  (* ASYNC_REG = "TRUE" *) logic [1:0] dci_s = '0;
  for (genvar i = 0; i < 3; i++) begin : g_dci
    DCIRESET u_dci (.RST(dci_rst), .LOCKED(dci_lk[i]));
  end
  always_ff @(posedge core_clk) begin
    if (core_rst) begin
      dci_cnt <= '0;
      dci_rst <= 1'b1;
    end else if (dci_cnt != ($clog2(DCI_RST_CYC) + 1)'(DCI_RST_CYC)) dci_cnt <= dci_cnt + 1'b1;
    else dci_rst <= 1'b0;
    if (dci_rst) dci_wt <= '0;
    else if (!dci_wt[$clog2(DCI_WAIT)]) dci_wt <= dci_wt + 1'b1;
    dci_s  <= {dci_s[0], &dci_lk};
    dci_ok <= !dci_rst && (dci_s[1] || dci_wt[$clog2(DCI_WAIT)]);
  end

  // ---- the DIMMs: each MIG in reset while the governor has it off (or DCI is not yet calibrated);
  // live UP_CYC cycles after it turns on (its AXI path and bridges out of reset: the MIG's
  // ui_clk runs by then), up once its MIG has calibrated (two flip-flops into core_clk)
  (* ASYNC_REG = "TRUE" *) logic [3:0] cal_c1 = '0, cal_c2 = '0;
  always_ff @(posedge core_clk) begin cal_c1 <= calib; cal_c2 <= cal_c1; end
  logic [3:0][$clog2(UP_CYC):0] up_cnt;
  logic [3:0] dimm_live = '0, dimm_up = '0;
  for (genvar d = 0; d < 4; d++) begin : g_dimm
    always_ff @(posedge core_clk) begin
      if (core_rst || !dimm_on[d] || !dci_ok) up_cnt[d] <= '0;
      else if (up_cnt[d] != ($clog2(UP_CYC) + 1)'(UP_CYC)) up_cnt[d] <= up_cnt[d] + 1'b1;
      mig_rst[d]   <= core_rst || !dimm_on[d] || !dci_ok;
      dimm_live[d] <= !core_rst && dimm_on[d] && up_cnt[d] == ($clog2(UP_CYC) + 1)'(UP_CYC);
      dimm_up[d]   <= dimm_live[d] && cal_c2[d];
    end
    // the DIMM's AXI path's reset in aclk (two flip-flops, then one more: sc_h's and the clock
    // converter's aresetn, the hmux's reset)
    (* ASYNC_REG = "TRUE" *) logic [1:0] sl = '0;
    logic sc_n = 1'b0;
    always_ff @(posedge aclk) begin sl <= {sl[0], dimm_live[d]}; sc_n <= sl[1]; end
    assign sc_rstn[d] = sc_n;
  end

  // ---- the accelerator's domain: acc_clk / acc_clk2x and acc_rst (as otpu_fpga_top_bcu)
  localparam int ACC_RST_CYC = 1024, ACC_STOP_CYC = 2048;
  typedef enum logic [1:0] {A_OFF, A_START, A_ON, A_STOP} acc_st_t;
  acc_st_t     acc_st = A_OFF;
  logic [11:0] acc_cnt = '0;
  logic        acc_rst0 = 1'b1, acc_up0 = 1'b0;
  logic        acc_rst_c;                   // acc_rst0 through its ACC_RC stages (core_clk, below)
  wire acc_want = acc_on && &dimm_ok && &dimm_up;
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
    // (with ACC_RA, on core_rst the clocks run on until the reset is through acc_rq: below)
    acc_ce  <= (acc_st != A_OFF && !core_rst) || (ACC_RA != 0 && !acc_rst_c);
    acc_rst0 <= acc_st != A_ON || core_rst;
    acc_up0  <= acc_st == A_ON && !core_rst;
  end
  // The accelerator's clocks: the MMCM's 2x output into two BUFGCE_DIVs on one enable and one
  // clear, acc_clk2x at /1 and acc_clk at /2 (UG949's pair: no MMCM phase error between them and
  // the jitter of one output, 0.182 ns less hold uncertainty on each 1x / 2x crossing than two
  // MMCM outputs; B1 placed: 94 % of the worst crossing's hold requirement was clock). acc_ce
  // (core_clk) straight into a CE had one 2x period (3.33 ns at 150 MHz) against the core clock
  // tree's ~3.3 ns over the buffer's input (f150a: -0.513 ns, its one failing path; UltraScale+
  // takes no CE_TYPE HARDSYNC). The enable is registered on a clock of its own instead, a BUFG of
  // the same output with one load (its root beside the buffers, the clocking wizard's safe-start
  // pattern), so both hops see a short tree. acc_ce changes after a core edge, an even 2x edge;
  // ce2x takes it at the odd edge after, and the buffers take their enable while their input is
  // low, so each run of acc_clk2x starts at an even edge (acc_clk rising with core_clk, as
  // Vivado's generated clock has it) and ends after an odd one: an even number of 2x periods, both
  // clocks stopped low and the divider where it started. clr2x clears the divider while both are
  // stopped (high from the 2x edge after a stop to the one that sets ce2x), so its phase does not
  // depend on configuration's start-up. Both clocks start and stop a core cycle after acc_ce
  // (acc_clk as with the wizard's enable buffer before), inside the sequencer's reset windows.
  logic ce2x_clk, ce2x = 1'b0, clr2x = 1'b1;
  BUFG u_ce2x_buf (.I(acc_clk2x_u), .O(ce2x_clk));
  always_ff @(posedge ce2x_clk) begin
    ce2x  <= acc_ce;
    clr2x <= !acc_ce && !ce2x;
  end
  BUFGCE_DIV #(.BUFGCE_DIVIDE(1), .SIM_DEVICE("ULTRASCALE_PLUS"))
    u_acc2x_buf (.I(acc_clk2x_u), .CE(ce2x), .CLR(clr2x), .O(acc_clk2x));
  BUFGCE_DIV #(.BUFGCE_DIVIDE(2), .SIM_DEVICE("ULTRASCALE_PLUS"))
    u_acc_buf (.I(acc_clk2x_u), .CE(ce2x), .CLR(clr2x), .O(acc_clk));
  // acc_rst and acc_up ACC_RC cycles later, together (the control path opens only once the
  // accelerator is out of reset). The reset drives the slice's ~2,200 synchronous resets in SLR2:
  // from one register it took 8.4 ns of route (dev100b, 0.16 ns of slack at 100 MHz). Its stages
  // let the placer carry it across the SLRs, and synthesis copies the last for every 256 loads.
  // ACC_RA (S 2: 2): the reset's last ACC_RA stages are acc_clk's (acc_rq). s2c150's acc_rst_p[2]
  // (core_clk, SLR1) straight into bridge core halves in SLR0 and SLR2 failed hold by 0.379 ns
  // (7 endpoints: acc_clk's leaves there 0.72-1.65 ns after core_clk's; setup 3.44 ns), and
  // neither phys_opt's hold fix nor route_design -auto_delay mended it. Now core_clk meets acc_clk
  // once, acc_rst_p[ACC_RP-1] into acc_rq[0] beside it in SLR1 (the floorplan's SLR1 pblock), and
  // acc_rq's last stage, a copy per 256 loads, goes out to the slices, the bridges and the control
  // in acc_clk alone. acc_up, and u_ctl_iso's acc_rst_c, take the same ACC_RA cycles more in
  // core_clk: u_sys's reset, acc_up and u_ctl_iso still change at one edge, ACC_RC cycles after
  // acc_rst0. acc_clk stops with acc_rq full of ones (A_STOP: ACC_STOP_CYC cycles of reset; on
  // core_rst acc_ce stays on until acc_rst_c, acc_rq[ACC_RA-1]'s twin, is up), so a restart sees
  // the reset from its first edge, as with core_clk's stages alone.
  localparam int ACC_RP = 3, ACC_RC = ACC_RP + ACC_RA;
  (* shreg_extract = "no", max_fanout = 256 *) logic [ACC_RC-1:0] acc_rst_p = '1;
  (* shreg_extract = "no" *)                   logic [ACC_RC-1:0] acc_up_p = '0;
  always_ff @(posedge core_clk) begin
    acc_rst_p <= {acc_rst_p[ACC_RC-2:0], acc_rst0};
    acc_up_p  <= {acc_up_p[ACC_RC-2:0], acc_up0};
  end
  assign acc_rst_c = acc_rst_p[ACC_RC-1];
  wire acc_up = acc_up_p[ACC_RC-1];
  logic acc_rst;                            // u_sys's (acc_clk with ACC_RA)
  if (ACC_RA == 0) begin : g_crq
    assign acc_rst = acc_rst_c;
  end else begin : g_arq
    (* shreg_extract = "no", max_fanout = 256 *) logic [ACC_RA-1:0] acc_rq = '1;
    always_ff @(posedge acc_clk) acc_rq <= ACC_RA'({acc_rq, acc_rst_p[ACC_RP-1]});
    assign acc_rst = acc_rq[ACC_RA-1];
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
    .clk(core_clk), .m_rst(core_rst), .s_rst(acc_rst_c),
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

  // ---- XDMA's DMA per DIMM behind otpu_axi_off: no new burst unless the DIMM is granted and up;
  // every burst answered by the gate once it is not live (its SmartConnect in reset: sc_h). Both in
  // core_clk, then two flip-flops into xdma_aclk. DIMM d takes window slot_of(d) (S 1, and S 2's
  // floorplans b and c: d's own)
  logic [3:0] xg_off = '1, xg_dead = '1;
  (* ASYNC_REG = "TRUE" *) logic [3:0] xo1 = '1, xo2 = '1, xd1 = '1, xd2 = '1;
  always_ff @(posedge xdma_aclk) begin xo1 <= xg_off; xo2 <= xo1; xd1 <= xg_dead; xd2 <= xd1; end
  for (genvar d = 0; d < 4; d++) begin : g_dma
    localparam int W = slot_of(d);
    always_ff @(posedge core_clk) begin
      xg_off[d]  <= !dimm_ok[d] || !dimm_up[d];
      xg_dead[d] <= !dimm_live[d];
    end
    otpu_axi_off #(.AW(DL), .DW(AXB), .NOUT(16)) u_xoff (
      .clk(xdma_aclk), .rst(xrst), .off(xo2[d]), .dead(xd2[d]),
      .m_awaddr(dm_awaddr[W][DL-1:0]), .m_awlen(dm_awlen[W]), .m_awvalid(dm_awvalid[W]),
      .m_awready(dm_awready[W]), .m_wdata(dm_wdata[W]), .m_wstrb(dm_wstrb[W]),
      .m_wlast(dm_wlast[W]), .m_wvalid(dm_wvalid[W]), .m_wready(dm_wready[W]),
      .m_bresp(dm_bresp[W]), .m_bvalid(dm_bvalid[W]), .m_bready(dm_bready[W]),
      .m_araddr(dm_araddr[W][DL-1:0]), .m_arlen(dm_arlen[W]), .m_arvalid(dm_arvalid[W]),
      .m_arready(dm_arready[W]), .m_rdata(dm_rdata[W]), .m_rresp(dm_rresp[W]),
      .m_rlast(dm_rlast[W]), .m_rvalid(dm_rvalid[W]), .m_rready(dm_rready[W]),
      .s_awaddr(hg_awaddr[d]), .s_awlen(hg_awlen[d]), .s_awvalid(hg_awvalid[d]),
      .s_awready(hg_awready[d]), .s_wdata(hg_wdata[d]), .s_wstrb(hg_wstrb[d]),
      .s_wlast(hg_wlast[d]), .s_wvalid(hg_wvalid[d]), .s_wready(hg_wready[d]),
      .s_bresp(hg_bresp[d]), .s_bvalid(hg_bvalid[d]), .s_bready(hg_bready[d]),
      .s_araddr(hg_araddr[d]), .s_arlen(hg_arlen[d]), .s_arvalid(hg_arvalid[d]),
      .s_arready(hg_arready[d]), .s_rdata(hg_rdata[d]), .s_rresp(hg_rresp[d]),
      .s_rlast(hg_rlast[d]), .s_rvalid(hg_rvalid[d]), .s_rready(hg_rready[d]));
  end

  // ---- the accelerator, its bridges and the DIMMs' pairs
  logic [2:0] board_led;
  // The port and the hub share acc_clk (never reset by RUN); Aurora's init clock is ungated.
  // Board reset/LRST comes back from otpu_ctrl as x2_rst. The governor may power-cycle the
  // accelerator domain: after doing so the host must program X2_CARD and LRST on both cards.
  logic [1:0][255:0] x2_s_tdata, x2_m_tdata;
  logic [1:0] x2_s_tvalid, x2_s_tlast, x2_s_tready;
  logic [1:0] x2_m_tvalid, x2_m_tlast, x2_m_thdr, x2_m_tready;
  logic [1:0] x2_up, x2_rx_err, x2_abort_in, x2_abort;
  logic x2_rst, x2_c_we;
  logic [7:0] x2_card, x2_ncards, x2_c_addr;
  logic [31:0] x2_c_wd, x2_c_rd;
`ifdef OTPU_WG_X2
  localparam int NCARDS = 2;
  wire [1:0] user_clk, tx_tvalid, tx_tready, rx_tvalid, channel_up, soft_err, unused_hard_err;
  wire [1:0][255:0] tx_tdata, rx_tdata;
  wire [1:0][3:0] lane_up;
  initial if (S != 2) $fatal(1, "otpu_fpga_top_bcu_wg: x2 needs two local slices");
  (* keep_hierarchy = "yes" *) otpu_link #(.CFG(`OTPU_WG_HASH)) u_link (
    .clk(acc_clk), .rst(x2_rst), .s_tdata(x2_s_tdata), .s_tvalid(x2_s_tvalid),
    .s_tlast(x2_s_tlast), .s_tready(x2_s_tready), .m_tdata(x2_m_tdata),
    .m_tvalid(x2_m_tvalid), .m_tlast(x2_m_tlast), .m_thdr(x2_m_thdr), .m_tready(x2_m_tready),
    .up(x2_up), .rx_err(x2_rx_err), .abort_in(x2_abort_in), .abort(x2_abort),
    .card(x2_card), .ncards(x2_ncards), .c_we(x2_c_we), .c_addr(x2_c_addr),
    .c_wd(x2_c_wd), .c_rd(x2_c_rd), .user_clk, .tx_tdata, .tx_tvalid, .tx_tready,
    .rx_tdata, .rx_tvalid, .channel_up, .lane_up, .soft_err);
  (* keep_hierarchy = "yes" *) otpu_link_phy_bcu u_link_phy (
    .init_clk, .rst(core_rst), .qsfp0_mgt_refclk_1_p, .qsfp0_mgt_refclk_1_n,
    .rxp({qsfp1_rx_p, qsfp0_rx_p}), .rxn({qsfp1_rx_n, qsfp0_rx_n}),
    .txp({qsfp1_tx_p, qsfp0_tx_p}), .txn({qsfp1_tx_n, qsfp0_tx_n}),
    .qsfp_resetl({qsfp1_resetl, qsfp0_resetl}), .qsfp_lpmode({qsfp1_lpmode, qsfp0_lpmode}),
    .qsfp_modsell({qsfp1_modsell, qsfp0_modsell}), .user_clk,
    .tx_tdata, .tx_tvalid, .tx_tready, .rx_tdata, .rx_tvalid,
    .channel_up, .lane_up, .soft_err, .hard_err(unused_hard_err));
`else
  localparam int NCARDS = 1;
  assign {x2_s_tready, x2_m_tdata, x2_m_tvalid, x2_m_tlast, x2_m_thdr,
          x2_up, x2_rx_err, x2_abort_in, x2_c_rd} = '0;
`endif
  // at S 2 the bridges' W rings' entry enables in 4 copies (WCP: B1's g_rs fu into 289 enables),
  // their read selects in 4 (RCP) and the pairs' W order's head in 4 (WOC: B1r's g_rs rp and
  // u_pair's u_wo head through the W muxes into the host muxes' W rings, -0.39 ns at 300 MHz),
  // and the pairs' R through a register a bridge (PRS: B1r's hmux R rings into the pairs' FIFOs)
  otpu_bcu_wg_sys #(.S(S), .NCH(8), .CB(128), .CAW(CAW), .CORE_KHZ(CORE_KHZ),
                    .CORE_KHZ1(CORE_KHZ1), .DDR_MTS(2400), .WCP(S == 2 ? 4 : 0),
                    .RCP(S == 2 ? 4 : 0), .WOC(S == 2 ? 4 : 1), .PRS(S == 2),
                    .RLY(RLY), .DMAP(DMAP), .BHW(BHW), .BHB(BHB), .NCARDS(NCARDS)) u_sys (
    .clk(acc_clk), .clk_f(acc_clk2x), .rst(acc_rst), .ctl_mrst(core_rst), .aclk,
    .arst(~dimm_live), .calib(dimm_up), .temp({device_temp, 2'b00}), .led(board_led),
    .x2_s_tdata, .x2_s_tvalid, .x2_s_tlast, .x2_s_tready,
    .x2_m_tdata, .x2_m_tvalid, .x2_m_tlast, .x2_m_thdr, .x2_m_tready,
    .x2_up, .x2_rx_err, .x2_abort_in, .x2_abort, .x2_rst, .x2_card, .x2_ncards,
    .x2_c_we, .x2_c_addr, .x2_c_wd, .x2_c_rd,
    .s_ctl_awaddr(ac_awaddr), .s_ctl_awvalid(ac_awvalid), .s_ctl_awready(ac_awready),
    .s_ctl_wdata(ac_wdata), .s_ctl_wstrb(ac_wstrb), .s_ctl_wvalid(ac_wvalid),
    .s_ctl_wready(ac_wready), .s_ctl_bresp(ac_bresp), .s_ctl_bvalid(ac_bvalid),
    .s_ctl_bready(ac_bready), .s_ctl_araddr(ac_araddr), .s_ctl_arvalid(ac_arvalid),
    .s_ctl_arready(ac_arready), .s_ctl_rdata(ac_rdata), .s_ctl_rresp(ac_rresp),
    .s_ctl_rvalid(ac_rvalid), .s_ctl_rready(ac_rready),
    .m_awvalid(p_awvalid), .m_awready(p_awready), .m_awid(p_awid), .m_awaddr(p_awaddr),
    .m_awlen(p_awlen), .m_wvalid(p_wvalid), .m_wready(p_wready), .m_wdata(p_wdata),
    .m_wstrb(p_wstrb), .m_wlast(p_wlast), .m_bvalid(p_bvalid), .m_bready(p_bready),
    .m_bid(p_bid), .m_bresp(p_bresp), .m_arvalid(p_arvalid), .m_arready(p_arready),
    .m_arid(p_arid), .m_araddr(p_araddr), .m_arlen(p_arlen), .m_rvalid(p_rvalid),
    .m_rready(p_rready), .m_rid(p_rid), .m_rdata(p_rdata), .m_rresp(p_rresp), .m_rlast(p_rlast));

  // ---- each DIMM's port: its pair and the host's DMA at aclk (M_AXI_H<d>) onto S_AXI_P<d>, reset
  // with the DIMM's AXI path (sc_rstn: an aclk register); at S 2 the W and R rings' read selects in
  // 8 copies (RCP: C0's beside slice 0 in SLR0, s2syn's R ring select -0.229 ns at 300 MHz), and
  // their entries' enables (WCP: B1's g_hmux[1] W ring fu into 577 enables)
  for (genvar d = 0; d < 4; d++) begin : g_hmux
    otpu_mig_hmux #(.IDW(4), .AW(MAW), .HB(S == 1 ? 2 : 0), .RCP(S == 2 ? 8 : 0),
                    .WCP(S == 2 ? 8 : 0)) u_hmux (.clk(aclk),
      .rst(!sc_rstn[d]),
      .p_awvalid(p_awvalid[d]), .p_awready(p_awready[d]), .p_awid(p_awid[d]),
      .p_awaddr(MAW'(p_awaddr[d])), .p_awlen(p_awlen[d]), .p_wvalid(p_wvalid[d]),
      .p_wready(p_wready[d]), .p_wdata(p_wdata[d]), .p_wstrb(p_wstrb[d]), .p_wlast(p_wlast[d]),
      .p_bvalid(p_bvalid[d]), .p_bready(p_bready[d]), .p_bid(p_bid[d]), .p_bresp(p_bresp[d]),
      .p_arvalid(p_arvalid[d]), .p_arready(p_arready[d]), .p_arid(p_arid[d]),
      .p_araddr(MAW'(p_araddr[d])), .p_arlen(p_arlen[d]), .p_rvalid(p_rvalid[d]),
      .p_rready(p_rready[d]), .p_rid(p_rid[d]), .p_rdata(p_rdata[d]), .p_rresp(p_rresp[d]),
      .p_rlast(p_rlast[d]),
      .h_awvalid(ha_awvalid[d]), .h_awready(ha_awready[d]), .h_awaddr(ha_awaddr[d]),
      .h_awlen(ha_awlen[d]), .h_wvalid(ha_wvalid[d]), .h_wready(ha_wready[d]),
      .h_wdata(ha_wdata[d]), .h_wstrb(ha_wstrb[d]), .h_wlast(ha_wlast[d]),
      .h_bvalid(ha_bvalid[d]), .h_bready(ha_bready[d]), .h_bresp(ha_bresp[d]),
      .h_arvalid(ha_arvalid[d]), .h_arready(ha_arready[d]), .h_araddr(ha_araddr[d]),
      .h_arlen(ha_arlen[d]), .h_rvalid(ha_rvalid[d]), .h_rready(ha_rready[d]),
      .h_rdata(ha_rdata[d]), .h_rresp(ha_rresp[d]), .h_rlast(ha_rlast[d]),
      .m_awvalid(sp_awvalid[d]), .m_awready(sp_awready[d]), .m_awid(sp_awid[d]),
      .m_awaddr(sp_awaddr[d]), .m_awlen(sp_awlen[d]), .m_wvalid(sp_wvalid[d]),
      .m_wready(sp_wready[d]), .m_wdata(sp_wdata[d]), .m_wstrb(sp_wstrb[d]), .m_wlast(sp_wlast[d]),
      .m_bvalid(sp_bvalid[d]), .m_bready(sp_bready[d]), .m_bid(sp_bid[d]), .m_bresp(sp_bresp[d]),
      .m_arvalid(sp_arvalid[d]), .m_arready(sp_arready[d]), .m_arid(sp_arid[d]),
      .m_araddr(sp_araddr[d]), .m_arlen(sp_arlen[d]), .m_rvalid(sp_rvalid[d]),
      .m_rready(sp_rready[d]), .m_rid(sp_rid[d]), .m_rdata(sp_rdata[d]), .m_rresp(sp_rresp[d]),
      .m_rlast(sp_rlast[d]));
  end

  // the pins a single-rank UDIMM's controller leaves: parity off, the other ranks idle
`define OTPU_WG_IDLE(N) \
  assign ddr4_c``N``_par = 1'b0; \
  assign ddr4_c``N``_cs_hi = 3'b111; \
  assign ddr4_c``N``_cke1 = 1'b0; \
  assign ddr4_c``N``_odt1 = 1'b0;
  `OTPU_WG_IDLE(0)
  `OTPU_WG_IDLE(1)
  `OTPU_WG_IDLE(2)
  `OTPU_WG_IDLE(3)
`undef OTPU_WG_IDLE
  assign uart_tx = 1'b1;

  // LEDs: led[0] the governor alive (1 Hz), led[1] PCIe link up and the governor in RUN, led[2]
  // the governor's WARN or a thermal trip, or the accelerator runs or halted cleanly
  assign led = {gov_led[2] | board_led[2] | board_led[1], pcie_link_up & gov_led[1], gov_led[0]};
endmodule
