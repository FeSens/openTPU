// The BCU1525 at WG 8 (targets/bcu1525-wg.toml; docs/bcu1525_wg.md): the accelerator (otpu_board,
// one slice of NCH = 8 channels of CB = 128 bytes, the WG targets') on the card's four DDR4 DIMMs, two
// virtual channels a DIMM. Channel j is an otpu_hbm_ch bridge (the WG targets', PC 0: its beat m at byte
// m x CB of its 2^CL bytes, CL = CAW + log2(CB); its AXI3 side at aclk, 256 bits) through a register
// slice (otpu_axi3_rs, NRS of them, R and B as plain stages: the bridges take every beat) into
// DIMM j >> 1's pair (otpu_mig_pair: channels 2d and 2d + 1 onto one 512-bit AXI4 port of CL + 1
// address bits, interleaved at 4 KiB inside the DIMM). The top takes each pair's port m_*[d]
// (aclk) into the block design: a SmartConnect with the host's DMA, across into the DIMM's MIG
// (ui_clk).
//
// S 2 (targets/bcu1525-s2.toml; docs/bcu1525_s2.md): two slices of NCH = 8
// channels each, a slice on two DIMMs, four of its channels a DIMM (otpu_mig_pair NC 4: the DIMM's
// byte {o[CL-1:12], q, o[11:0]} is its quarter q's byte o). Board channel g (slice g / NCH's channel
// g mod NCH) is in DIMM slot g / NC (NC = 2 S channels a DIMM), slot i on DIMM DMAP[2 i +: 2]: the
// slices' DIMMs, a permutation of the four (the floorplan's choice: slice s on slots 2 s and
// 2 s + 1). At S 1 slot i is DIMM i (DMAP 8'hE4) as before. m_*[d] is DIMM d's port whatever the
// map, so the top and the block design do not change with it; calib[d] and arst[d] are DIMM d's.
//
// CAW, the channels' beat address bits: 22 (512 MiB a channel, 4 GiB the card: the first builds),
// 25 (4 GiB a channel, an 8 GB DIMM's two: the 32 GB card) or 26 (64 GB: 16 GB DIMMs, later); at
// S 2 a channel is a quarter DIMM: 24 the 32 GB card's (2 GiB, 16 GiB a slice), 25 64 GB's.
// otpu_board takes DRAM_LOG2 (the card's bytes) and CH_LOG2 (a channel's) from it here, never
// from OTPU_BOARD (opentpu.host.boardmacro: DERIVED); its AW (the address registers' bits,
// at least DRAM_LOG2) comes in OTPU_BOARD.
//
// otpu_board's parameters other than this module's own (S, the channels, the clock, DDR_MTS, no
// I2C, no host calibration, the BMC) are OTPU_BOARD's, one list of ", .K(V)" overrides (opentpu.host.boardmacro
// board_defines: every otpu_board parameter of the target's rtl_params; boards/
// bcu1525/scripts/make_wg.py writes them for the build, opentpu/host/board.py for the board
// model). Without it the module stops (otpu_board's defaults are no target's).
//
// NAT_XPOST: the bridges post port W's beats into the MXU's chunk FIFO a quarter (256 bits, an AXI
// beat) at a time on aclk (p_clk, NAT_XQB 2), as the WG targets'.
//
// Clocks: clk (the accelerator's, CORE_KHZ), clk_f (2 x clk, rising with it: TMEM_PMP 1), aclk
// (the bridges' AXI side and the pairs', 300 MHz: a bridge's 256 bits move 9.6 GB/s, half a
// DDR4-2400 DIMM). arst[d]: DIMM d's port in reset (any clock; the top's: the DIMM off,
// uncalibrated or its SmartConnect in reset), through a synchronizer a DIMM into its bridges' AXI
// halves, their register slices and its pair. calib[d]: DIMM d calibrated (any clock; otpu_board
// synchronizes it: STATUS CALIB). RLY[4 j +: 4]: channel j's bridge pipeline between its halves
// (otpu_hbm_ch N; the DIMMs a crossing or two from the slice), a far channel's W at half width.
`ifndef OTPU_BOARD
`define OTPU_BOARD
`define OTPU_BOARD_UNSET
`endif
`ifndef OTPU_WPOST                // the board's NAT_WPOST (opentpu.host.boardmacro board_defines)
`define OTPU_WPOST 0
`endif
`ifndef OTPU_XPOST                // the board's NAT_XPOST
`define OTPU_XPOST 0
`endif
`ifndef OTPU_RHB                  // the board's NAT_RHB / NAT_WHB
`define OTPU_RHB 0
`endif
`ifndef OTPU_WHB
`define OTPU_WHB 0
`endif
`ifndef OTPU_XSW                  // log2 of the board's FIFO_DEPTH
`define OTPU_XSW 10
`endif
module otpu_bcu_wg_sys #(
  parameter int NCARDS   = 1,           // cards; link ends exposed to the FPGA top
  parameter int S        = 1,             // slices (1 or 2), each on NCH channels
  parameter int NCH      = 8,
  parameter int CB       = 128,
  parameter int CAW      = 22,
  parameter int CORE_KHZ = 100000,
  parameter int CORE_KHZ1 = CORE_KHZ,     // the fallback clock (otpu_clk_k), or CORE_KHZ: none
  parameter int DDR_MTS  = 2400,
  parameter logic [4*S*NCH-1:0] RLY = '0,
  parameter logic [7:0]  DMAP     = 8'hE4, // DIMM slot i's DIMM, bits 2 i +: 2 (S 2: slice s on
                                          //   slots 2 s, 2 s + 1; E4: slot i on DIMM i)
  parameter int NRS      = 1,             // register slices between a bridge and its pair
  parameter int WCP      = 0,             // their W rings' entry enables in copies (otpu_rs WCP)
  parameter int RCP      = 0,             // and their read selects (otpu_rs RCP)
  parameter int WOC      = 1,             // the pairs' W order's head in copies (otpu_mig_pair WOC)
  parameter bit PRS      = 1'b0,          // the pairs' R through a register a bridge (otpu_mig_pair RS)
  parameter int MWAIT    = 8,             // aclk cycles a partial burst waits (otpu_hbm_ch mwait)
  parameter int ARD      = 32,            // the bridges' read credits (with NAT_XPOST, the WG targets')
  parameter int PRD      = 32,            // the pairs' R FIFOs, 512-bit beats (otpu_mig_pair RD)
  parameter int BHW      = 0,             // the pairs' bank-half window (otpu_mig_pair: a channel's
  //                                         first 2^BHW bytes; 0: none; CAPS2's the slice's)
  parameter int BHB      = 16             // and the MIG map's bank-high bit
) (
  // Two link ports on the core clock (otpu_link in the FPGA top). All outputs are tied
  // off at NC 1. x2_rst resets the link and the hub's link side independently of RUN.
  output logic [1:0][255:0] x2_s_tdata,
  output logic [1:0]        x2_s_tvalid, x2_s_tlast,
  input  logic [1:0]        x2_s_tready,
  input  logic [1:0][255:0] x2_m_tdata,
  input  logic [1:0]        x2_m_tvalid, x2_m_tlast, x2_m_thdr,
  output logic [1:0]        x2_m_tready,
  input  logic [1:0]        x2_up, x2_rx_err, x2_abort_in,
  output logic [1:0]        x2_abort,
  output logic             x2_rst,
  output logic [7:0]       x2_card, x2_ncards,
  output logic             x2_c_we,
  output logic [7:0]       x2_c_addr,
  output logic [31:0]      x2_c_wd,
  input  logic [31:0]      x2_c_rd,
  input  logic                    clk,
  input  logic                    clk_f,
  input  logic                    rst,
  input  logic                    ctl_mrst,     // the control master's reset, in clk (otpu_board)
  input  logic                    aclk,
  input  logic [NCH/2-1:0]        arst,
  input  logic [NCH/2-1:0]        calib,
  input  logic [11:0]             temp,
  output logic [2:0]              led,
  // control registers: AXI4-Lite slave, clk (BAR0 0x0)
  input  logic [11:0]             s_ctl_awaddr,
  input  logic                    s_ctl_awvalid,
  output logic                    s_ctl_awready,
  input  logic [31:0]             s_ctl_wdata,
  input  logic [3:0]              s_ctl_wstrb,
  input  logic                    s_ctl_wvalid,
  output logic                    s_ctl_wready,
  output logic [1:0]              s_ctl_bresp,
  output logic                    s_ctl_bvalid,
  input  logic                    s_ctl_bready,
  input  logic [11:0]             s_ctl_araddr,
  input  logic                    s_ctl_arvalid,
  output logic                    s_ctl_arready,
  output logic [31:0]             s_ctl_rdata,
  output logic [1:0]              s_ctl_rresp,
  output logic                    s_ctl_rvalid,
  input  logic                    s_ctl_rready,
  // the DIMMs' ports (otpu_mig_pair m_*: AXI4 512-bit, aclk; AxSIZE 64 bytes, INCR: the top's)
  output logic [NCH/2-1:0]        m_awvalid,
  input  logic [NCH/2-1:0]        m_awready,
  output logic [NCH/2-1:0][3:0]   m_awid,
  output logic [NCH/2-1:0][CAW+$clog2(CB)+$clog2(2*S)-1:0] m_awaddr,
  output logic [NCH/2-1:0][7:0]   m_awlen,
  output logic [NCH/2-1:0]        m_wvalid,
  input  logic [NCH/2-1:0]        m_wready,
  output logic [NCH/2-1:0][511:0] m_wdata,
  output logic [NCH/2-1:0][63:0]  m_wstrb,
  output logic [NCH/2-1:0]        m_wlast,
  input  logic [NCH/2-1:0]        m_bvalid,
  output logic [NCH/2-1:0]        m_bready,
  input  logic [NCH/2-1:0][3:0]   m_bid,
  input  logic [NCH/2-1:0][1:0]   m_bresp,
  output logic [NCH/2-1:0]        m_arvalid,
  input  logic [NCH/2-1:0]        m_arready,
  output logic [NCH/2-1:0][3:0]   m_arid,
  output logic [NCH/2-1:0][CAW+$clog2(CB)+$clog2(2*S)-1:0] m_araddr,
  output logic [NCH/2-1:0][7:0]   m_arlen,
  input  logic [NCH/2-1:0]        m_rvalid,
  output logic [NCH/2-1:0]        m_rready,
  input  logic [NCH/2-1:0][3:0]   m_rid,
  input  logic [NCH/2-1:0][511:0] m_rdata,
  input  logic [NCH/2-1:0][1:0]   m_rresp,
  input  logic [NCH/2-1:0]        m_rlast
);
  localparam int ND = NCH / 2;            // DIMMs (S NCH channels, NC a DIMM)
  localparam int NC = 2 * S;              // channels a DIMM
  localparam int NT = S * NCH;            // the board's channels
  localparam int XSW = `OTPU_XSW;
  localparam int CL  = CAW + $clog2(CB);  // a channel's byte address bits (a DIMM's CL + log2 NC)
  // DIMM slot i's DIMM (DMAP), and the DIMM of board channel g
  function automatic int slot_dimm(input int i);
    return int'(DMAP[2 * i +: 2]);
  endfunction
  function automatic int ch_dimm(input int g);
    return slot_dimm(g / NC);
  endfunction
  function automatic bit dmap_ok();
    logic [3:0] seen;
    seen = '0;
    for (int i = 0; i < ND; i++) begin
      if (slot_dimm(i) >= ND || seen[slot_dimm(i)]) return 1'b0;
      seen[slot_dimm(i)] = 1'b1;
    end
    return 1'b1;
  endfunction
  initial if (NCH % 2 != 0 || CB != 128 || CAW < 22 || CAW > 26 || S < 1 || S > 2 || ND > 4 ||
              (S == 2 && NCH != 8) || !dmap_ok())
    $fatal(1, "otpu_bcu_wg_sys: S %0d, %0d channels of %0d bytes a slice (two 128-byte channels a DIMM; S 2: 8, four a DIMM), %0d-bit beat addresses (22 .. 26), DMAP %h (a permutation of the %0d DIMMs)",
           S, NCH, CB, CAW, DMAP, ND);
`ifdef OTPU_BOARD_UNSET
  initial $fatal(1, "otpu_bcu_wg_sys: no OTPU_BOARD (otpu_board's parameters)");
`endif

  // ---- the accelerator's native ports and posted beats
  logic [NT-1:0]            n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid, n_err;
  logic [NT-1:0][5:0]       n_wfree;
  logic [NT-1:0][CAW-1:0]   n_caddr;
  logic [NT-1:0][CB*8-1:0]  n_wdata, n_rdata;
  logic [NT-1:0][CB-1:0]    n_wmask;
  logic [NT-1:0][15:0]      n_wdone;
  logic [NT-1:0][1:0]       n_csrc;
  logic [NT-1:0][XSW-1:0]   n_cslot;
  logic [NT-1:0][4:0][XSW-1:0] p_slot;
  logic [NT-1:0]            p_we;
  logic [NT-1:0][1:0][CB/32-1:0] p_q;
  logic [NT-1:0][255:0]     p_d;
  logic [NT-1:0][XSW+1:0]   p_cntg;
  logic [NT-1:0]            ch_calib;
  for (genvar j = 0; j < NT; j++) begin : g_cal
    assign ch_calib[j] = calib[ch_dimm(j)];
  end

  otpu_board #(.NC(NCARDS), .S(S), .NCH(NCH), .CB(CB), .CORE_KHZ(CORE_KHZ), .CORE_KHZ1(CORE_KHZ1),
               .DDR_MTS(DDR_MTS),
               .HAS_I2C(1'b0), .HOSTCAL(1'b0), .HAS_BMC(1'b1), .NAT_XQB(2),
               .BHW(BHW != 0 ? BHW + $clog2(NCH) : 0),
               .DRAM_LOG2(CAW + $clog2(NCH * CB)), .CH_LOG2(CL) `OTPU_BOARD) u_board (
    .clk, .clk_f, .rst, .ctl_mrst, .calib(ch_calib), .ded('0), .temp, .led, .i2c_lo(),
    .i2c_pin(5'h1F),
    .x2_s_tdata, .x2_s_tvalid, .x2_s_tlast, .x2_s_tready,
    .x2_m_tdata, .x2_m_tvalid, .x2_m_tlast, .x2_m_thdr, .x2_m_tready,
    .x2_up, .x2_rx_err, .x2_abort_in, .x2_abort, .x2_rst, .x2_card, .x2_ncards,
    .x2_c_we, .x2_c_addr, .x2_c_wd, .x2_c_rd,
    .s_ctl_awaddr, .s_ctl_awvalid, .s_ctl_awready, .s_ctl_wdata, .s_ctl_wstrb, .s_ctl_wvalid,
    .s_ctl_wready, .s_ctl_bresp, .s_ctl_bvalid, .s_ctl_bready, .s_ctl_araddr, .s_ctl_arvalid,
    .s_ctl_arready, .s_ctl_rdata, .s_ctl_rresp, .s_ctl_rvalid, .s_ctl_rready,
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask,
    .n_rvalid, .n_rdata, .n_wdone, .n_wfree, .n_err,
    .n_csrc, .n_cslot, .p_clk(aclk), .p_we, .p_q, .p_slot, .p_d, .p_cntg);

  // ---- each DIMM's reset in aclk: two registers (their input a false path), then a register
  // for each load (wrs_b: a bridge's AXI half, wrs_s: its register slices, wrs_p: the pair), so
  // each starts beside its own (KEEP: synthesis would merge them) and phys_screp.tcl
  // (OTPU_WG_PREPHYS) copies the wide ones by placement. B1r: g_ars[0].s_reg[1] into the pair's
  // FIFO pointers and the slices' rings, -0.39 ns at 300 MHz, ~97 % route. One cycle more for all
  // of a DIMM's loads alike
  logic [ND-1:0] dars;
  for (genvar d = 0; d < ND; d++) begin : g_ars
    (* ASYNC_REG = "TRUE" *) logic [1:0] s = 2'b11;
    always_ff @(posedge aclk) s <= {s[0], arst[d]};
    assign dars[d] = s[1];
  end

  // ---- the bridges, their register slices (b_*: the pairs' side); at S 2 the core halves' aclk
  // fan-outs in copies (otpu_hbm_ch CPY)
  logic [NT-1:0]          q_awvalid, q_awready, q_wvalid, q_wready, q_wlast, q_bvalid, q_bready;
  logic [NT-1:0]          q_arvalid, q_arready, q_rvalid, q_rready, q_rlast;
  logic [NT-1:0][33:0]    q_awaddr, q_araddr;
  logic [NT-1:0][3:0]     q_awlen, q_arlen;
  logic [NT-1:0][255:0]   q_wdata, q_rdata;
  logic [NT-1:0][31:0]    q_wstrb;
  logic [NT-1:0][1:0]     q_bresp, q_rresp;
  logic [NT-1:0]          b_awvalid, b_awready, b_wvalid, b_wready, b_wlast, b_bvalid, b_bready;
  logic [NT-1:0]          b_arvalid, b_arready, b_rvalid, b_rready, b_rlast;
  logic [NT-1:0][33:0]    b_awaddr, b_araddr;
  logic [NT-1:0][3:0]     b_awlen, b_arlen;
  logic [NT-1:0][255:0]   b_wdata, b_rdata;
  logic [NT-1:0][31:0]    b_wstrb;
  logic [NT-1:0][1:0]     b_bresp, b_rresp;
  for (genvar j = 0; j < NT; j++) begin : g_ch
    (* keep = "true", shreg_extract = "no" *) logic wrs_b = 1'b1, wrs_s = 1'b1;
    always_ff @(posedge aclk) begin
      wrs_b <= dars[ch_dimm(j)];
      wrs_s <= dars[ch_dimm(j)];
    end
    localparam int N = int'(RLY[4 * j +: 4]);
    (* keep_hierarchy = "yes" *)
    otpu_hbm_ch #(.CB(CB), .CAW(CAW), .PC(5'd0), .OAW(34), .ARD(ARD), .WDD(32), .WPOST(`OTPU_WPOST),
                  .N(N), .WH(N != 0), .AWD(32), .PST(1), .XSW(XSW), .RHB(`OTPU_RHB),
                  .WHB(`OTPU_WHB), .BHW(BHW), .CPY(S == 2)) u_br (.clk, .rst,
      .n_cvalid(n_cvalid[j]), .n_cready(n_cready[j]), .n_cwe(n_cwe[j]), .n_caddr(n_caddr[j]),
      .n_wvalid(n_wvalid[j]), .n_wready(n_wready[j]), .n_wfree(n_wfree[j]), .n_wdata(n_wdata[j]),
      .n_wmask(n_wmask[j]),
      .n_rvalid(n_rvalid[j]), .n_rdata(n_rdata[j]), .n_wdone(n_wdone[j]), .n_err(n_err[j]),
      .n_csrc(n_csrc[j]), .n_cslot(n_cslot[j]), .p_we(p_we[j]), .p_q(p_q[j]), .p_slot(p_slot[j]),
      .p_d(p_d[j]), .p_cntg(p_cntg[j]),
      .aclk, .arst(wrs_b), .mwait(6'(MWAIT)),
      .m_awvalid(q_awvalid[j]), .m_awready(q_awready[j]), .m_awaddr(q_awaddr[j]), .m_awlen(q_awlen[j]),
      .m_wvalid(q_wvalid[j]), .m_wready(q_wready[j]), .m_wdata(q_wdata[j]), .m_wstrb(q_wstrb[j]),
      .m_wlast(q_wlast[j]), .m_bvalid(q_bvalid[j]), .m_bready(q_bready[j]), .m_bresp(q_bresp[j]),
      .m_arvalid(q_arvalid[j]), .m_arready(q_arready[j]), .m_araddr(q_araddr[j]), .m_arlen(q_arlen[j]),
      .m_rvalid(q_rvalid[j]), .m_rready(q_rready[j]), .m_rdata(q_rdata[j]), .m_rresp(q_rresp[j]),
      .m_rlast(q_rlast[j]), .ev(), .ev_n());
    if (NRS > 0) begin : g_rs
      otpu_axi3_rs #(.N(NRS), .RBP(1'b1), .WCP(WCP), .RCP(RCP)) u_rs (.clk(aclk), .rst(wrs_s),
        .s_awvalid(q_awvalid[j]), .s_awready(q_awready[j]), .s_awid(6'd0), .s_awaddr(q_awaddr[j]),
        .s_awlen(q_awlen[j]), .s_wvalid(q_wvalid[j]), .s_wready(q_wready[j]), .s_wdata(q_wdata[j]),
        .s_wstrb(q_wstrb[j]), .s_wlast(q_wlast[j]), .s_bvalid(q_bvalid[j]), .s_bready(q_bready[j]),
        .s_bid(), .s_bresp(q_bresp[j]), .s_arvalid(q_arvalid[j]), .s_arready(q_arready[j]),
        .s_arid(6'd0), .s_araddr(q_araddr[j]), .s_arlen(q_arlen[j]), .s_rvalid(q_rvalid[j]),
        .s_rready(q_rready[j]), .s_rid(), .s_rdata(q_rdata[j]), .s_rresp(q_rresp[j]),
        .s_rlast(q_rlast[j]),
        .m_awvalid(b_awvalid[j]), .m_awready(b_awready[j]), .m_awid(), .m_awaddr(b_awaddr[j]),
        .m_awlen(b_awlen[j]), .m_wvalid(b_wvalid[j]), .m_wready(b_wready[j]), .m_wdata(b_wdata[j]),
        .m_wstrb(b_wstrb[j]), .m_wlast(b_wlast[j]), .m_bvalid(b_bvalid[j]), .m_bready(b_bready[j]),
        .m_bid(6'd0), .m_bresp(b_bresp[j]), .m_arvalid(b_arvalid[j]), .m_arready(b_arready[j]),
        .m_arid(), .m_araddr(b_araddr[j]), .m_arlen(b_arlen[j]), .m_rvalid(b_rvalid[j]),
        .m_rready(b_rready[j]), .m_rid(6'd0), .m_rdata(b_rdata[j]), .m_rresp(b_rresp[j]),
        .m_rlast(b_rlast[j]));
    end else begin : g_dir
      assign b_awvalid[j] = q_awvalid[j]; assign q_awready[j] = b_awready[j];
      assign b_awaddr[j] = q_awaddr[j]; assign b_awlen[j] = q_awlen[j];
      assign b_wvalid[j] = q_wvalid[j]; assign q_wready[j] = b_wready[j];
      assign b_wdata[j] = q_wdata[j]; assign b_wstrb[j] = q_wstrb[j]; assign b_wlast[j] = q_wlast[j];
      assign q_bvalid[j] = b_bvalid[j]; assign b_bready[j] = q_bready[j]; assign q_bresp[j] = b_bresp[j];
      assign b_arvalid[j] = q_arvalid[j]; assign q_arready[j] = b_arready[j];
      assign b_araddr[j] = q_araddr[j]; assign b_arlen[j] = q_arlen[j];
      assign q_rvalid[j] = b_rvalid[j]; assign b_rready[j] = q_rready[j];
      assign q_rdata[j] = b_rdata[j]; assign q_rresp[j] = b_rresp[j]; assign q_rlast[j] = b_rlast[j];
    end
  end

  // ---- the DIMMs' pairs (at S 2 of four channels): slot i's channels NC i .. NC i + NC - 1 on
  // DIMM d = DMAP[2 i +: 2] (S 1: channels 2d and 2d + 1)
  for (genvar i = 0; i < ND; i++) begin : g_dimm
    localparam int d = slot_dimm(i);
    logic [NC-1:0] bv;
    logic [1:0]    br;
    (* keep = "true", shreg_extract = "no" *) logic wrs_p = 1'b1;
    always_ff @(posedge aclk) wrs_p <= dars[d];
    (* keep_hierarchy = "yes" *)
    otpu_mig_pair #(.IDW(4), .RD(PRD), .CL(CL), .NC(NC), .BHW(BHW), .BHB(BHB), .WOC(WOC),
                    .RS(PRS)) u_pair (
      .clk(aclk),
      .rst(wrs_p),
      .s_awvalid(b_awvalid[NC*i +: NC]), .s_awready(b_awready[NC*i +: NC]),
      .s_awaddr(b_awaddr[NC*i +: NC]), .s_awlen(b_awlen[NC*i +: NC]),
      .s_wvalid(b_wvalid[NC*i +: NC]), .s_wready(b_wready[NC*i +: NC]),
      .s_wdata(b_wdata[NC*i +: NC]), .s_wstrb(b_wstrb[NC*i +: NC]), .s_wlast(b_wlast[NC*i +: NC]),
      .s_bvalid(bv), .s_bready(b_bready[NC*i +: NC]), .s_bresp(br),
      .s_arvalid(b_arvalid[NC*i +: NC]), .s_arready(b_arready[NC*i +: NC]),
      .s_araddr(b_araddr[NC*i +: NC]), .s_arlen(b_arlen[NC*i +: NC]),
      .s_rvalid(b_rvalid[NC*i +: NC]), .s_rready(b_rready[NC*i +: NC]),
      .s_rdata(b_rdata[NC*i +: NC]), .s_rresp(b_rresp[NC*i +: NC]), .s_rlast(b_rlast[NC*i +: NC]),
      .m_awvalid(m_awvalid[d]), .m_awready(m_awready[d]), .m_awid(m_awid[d]), .m_awaddr(m_awaddr[d]),
      .m_awlen(m_awlen[d]), .m_wvalid(m_wvalid[d]), .m_wready(m_wready[d]), .m_wdata(m_wdata[d]),
      .m_wstrb(m_wstrb[d]), .m_wlast(m_wlast[d]), .m_bvalid(m_bvalid[d]), .m_bready(m_bready[d]),
      .m_bid(m_bid[d]), .m_bresp(m_bresp[d]), .m_arvalid(m_arvalid[d]), .m_arready(m_arready[d]),
      .m_arid(m_arid[d]), .m_araddr(m_araddr[d]), .m_arlen(m_arlen[d]), .m_rvalid(m_rvalid[d]),
      .m_rready(m_rready[d]), .m_rid(m_rid[d]), .m_rdata(m_rdata[d]), .m_rresp(m_rresp[d]),
      .m_rlast(m_rlast[d]));
    assign b_bvalid[NC*i +: NC] = bv;
    assign b_bresp[NC*i +: NC] = {NC{br}};
  end
endmodule
