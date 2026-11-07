// The accelerator and XDMA's DMA on the SQRL FK35's HBM (otpu_fpga_top_fk35: one otpu_hbm_port per channel on
// the HBM IP's AXI ports): otpu_board with S = 2 slices (DUAL: slice s on board channels 4s .. 4s + 3,
// otpu_native_dram2 each; else on 2s and 2s + 1, otpu_native_dram) and one otpu_mem_ch per channel in front of that
// channel's controller ports c_* in uclk (the adapters' and the HBM AXI ports' clock), meeting XDMA's master for that
// channel there (the block design's SmartConnect sc_dma splits XDMA's DMA by card address bits from 31 up, channel j
// at j * 2 GiB, and converts it to 128 bits). A copy of the
// BCU1525's otpu_bcu_sys for this board: its two SLRs put the HBM, XDMA and the channels in SLR0 and otpu_board (both
// slices and the shell) in SLR1, so every channel's native port crosses (SLL; boards/sqrl-fk35/constraints/
// fk35_floorplan.tcl).
//   calib: each channel's calibration flag in its controller's clock (STATUS CALIB0 / CALIB1: the even /
//        odd channels; synchronized in otpu_board). ded: each channel's ECC has
//        counted an uncorrectable word (STATUS ECC_DED; the same).
// The parameters are the target's (targets/sqrl-fk35.toml through opentpu.target rtl_params; the defaults are its
// values, tests/test_target.py).
module otpu_fk35_sys #(
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
  parameter int TMEM_IMPL  = 0,           // TMEM: 0 a copy per read port, 3 one copy (otpu_board)
  parameter bit TMEM_RQ    = 1'b0,
  parameter int TMEM_NSB   = 8,             // TMEM_IMPL 3 (one copy, docs/tmem_nocopy.md; otpu_board):
  parameter int TMEM_SBP   = 3,             //   RAMs a bank, the sub-bank's address bit, its hash,
  parameter int TMEM_SBH   = 0,             //   replays, true dual-port RAMs, QUANT 3 and COLL the
  parameter bit TMEM_RPL   = 1'b0,          //   DMA's guests, the RAMs pumped at 2x on clk_f (1),
  parameter bit TMEM_TDP   = 1'b0,          //   the read ports' ranks and RAM sides
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
  parameter int CORE_KHZ   = 250000,
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int DDR_MTS    = 1800,
  parameter bit SEQ_WC     = 1'b0,    // EXPERIMENTAL: otpu_seq's wait entries (WAITC, docs/seq_vpu.md; CFG_UARCH bit24)
  parameter int PLATFORM   = 4,
  parameter logic [31:0] ARCH_HASH = 32'h0,
  parameter bit DUAL       = 1'b1,          // every slice on four channels, port B pairs (otpu_board DUAL)
  parameter bit QUAD       = 1'b0,          // (DUAL) 4-bit MM PAIR four blocks an advance (otpu_board QUAD)
  parameter int NQMEM      = 0,             // the adapters' queue memories (otpu_native_dram QMEM)
  parameter bit MEMX       = 1'b0,          // the adapters in SLR0 behind a crossing of the slices' ports
                                            // (otpu_board MEMX): the native ports then stay in SLR0
  parameter int NC         = (DUAL ? 4 : 2) * S,  // memory channels (two per slice; DUAL four)
  // bit j: board channel j's native port crosses an SLR (both slices in SLR1, every channel in SLR0 with the HBM:
  // boards/sqrl-fk35/constraints/fk35_floorplan.tcl): two otpu_native_reg, u_tx with the slice and u_rx with the
  // channel, flip-flop to flip-flop across; none with MEMX
  parameter logic [NC-1:0] SLL = {NC{!MEMX}}
) (
  // core clock and reset (synchronous, high), XDMA's clock and reset
  input  logic                    clk,
  input  logic                    clk_f,        // TMEM_PMP 1: 2 x clk, rising with it (otpu_board)
  input  logic                    rst,
  input  logic                    ctl_mrst,     // the control master's reset, in clk (otpu_board)
  input  logic                    xclk,
  input  logic                    xrst,
  // board status and pins (otpu_board)
  input  logic [NC-1:0]           calib,
  input  logic [NC-1:0]           ded,
  input  logic [11:0]             temp,
  output logic [2:0]              led,
  output logic [3:0]              i2c_lo,
  input  logic [4:0]              i2c_pin,
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
  // XDMA's DMA, one AXI4 master per channel: 128 bits, xclk, no IDs (answered in order), INCR
  // bursts of full-width beats; the channel's 2 GiB in address bits 30:0
  input  logic [NC-1:0]           x_awvalid,
  output logic [NC-1:0]           x_awready,
  input  logic [NC-1:0][30:0]     x_awaddr,
  input  logic [NC-1:0][7:0]      x_awlen,
  input  logic [NC-1:0]           x_wvalid,
  output logic [NC-1:0]           x_wready,
  input  logic [NC-1:0][127:0]    x_wdata,
  input  logic [NC-1:0][15:0]     x_wstrb,
  input  logic [NC-1:0]           x_wlast,
  output logic [NC-1:0]           x_bvalid,
  input  logic [NC-1:0]           x_bready,
  output logic [NC-1:0][1:0]      x_bresp,
  input  logic [NC-1:0]           x_arvalid,
  output logic [NC-1:0]           x_arready,
  input  logic [NC-1:0][30:0]     x_araddr,
  input  logic [NC-1:0][7:0]      x_arlen,
  output logic [NC-1:0]           x_rvalid,
  input  logic [NC-1:0]           x_rready,
  output logic [NC-1:0][127:0]    x_rdata,
  output logic [NC-1:0][1:0]      x_rresp,
  output logic [NC-1:0]           x_rlast,
  // the channels' controller ports, channel j in uclk[j], two ports each ([channel][port];
  // otpu_mem_ch's c_*: port p takes the banks of parity p; the beat within the 2 GiB a slice
  // addresses, 25 bits: otpu_hbm_port's c_cmd_addr)
  input  logic [NC-1:0]           uclk,
  input  logic [NC-1:0]           urst,
  output logic [NC-1:0][1:0]      c_cmd_valid,
  input  logic [NC-1:0][1:0]      c_cmd_ready,
  output logic [NC-1:0][1:0]      c_cmd_we,
  output logic [NC-1:0][1:0][24:0] c_cmd_addr,
  output logic [NC-1:0][1:0]      c_wdata_valid,
  input  logic [NC-1:0][1:0]      c_wdata_ready,
  output logic [NC-1:0][1:0][511:0] c_wdata_data,
  output logic [NC-1:0][1:0][63:0] c_wdata_we,
  input  logic [NC-1:0][1:0]      c_rdata_valid,
  input  logic [NC-1:0][1:0][511:0] c_rdata_data
);
  // the accelerator's native masters, per channel (the board's side: b_*; the bridge's: n_*)
  logic [NC-1:0]        b_cvalid, b_cready, b_cwe, b_wvalid, b_wready, b_rvalid, b_err;
  logic [NC-1:0][24:0]  b_caddr;
  logic [NC-1:0][511:0] b_wdata, b_rdata;
  logic [NC-1:0][63:0]  b_wmask;
  logic [NC-1:0][15:0]  b_wdone;
  logic [NC-1:0]        n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid, n_err;
  logic [NC-1:0][24:0]  n_caddr;
  logic [NC-1:0][511:0] n_wdata, n_rdata;
  logic [NC-1:0][63:0]  n_wmask;
  logic [NC-1:0][15:0]  n_wdone;
  // the core reset, registered once on each side of the SLR crossing: rst_b for otpu_board and the u_tx halves (SLR1),
  // rst_c[j] for channel j's otpu_mem_ch and u_rx (SLR0), so that no reset net crosses from a flip-flop to its loads
  // in both SLRs; KEEP stops synthesis merging the equivalent copies
  (* KEEP = "TRUE" *) logic rst_b;
  (* KEEP = "TRUE" *) logic [NC-1:0] rst_c;
  always_ff @(posedge clk) begin
    rst_b <= rst;
    rst_c <= {NC{rst}};
  end

  for (genvar j = 0; j < NC; j++) begin : g_ch
    if (SLL[j]) begin : g_sll
      logic cv, cr, cwe, wv, wr, rv, err;
      logic [24:0] ca;
      logic [511:0] wd, rd;
      logic [63:0] wm;
      logic [15:0] wdone;
      // kept as cells: the floorplan puts u_tx in SLR1 and u_rx in SLR0 by name
      (* KEEP_HIERARCHY = "YES" *)
      otpu_native_reg u_tx (.clk, .rst(rst_b),
        .s_cvalid(b_cvalid[j]), .s_cready(b_cready[j]), .s_cwe(b_cwe[j]), .s_caddr(b_caddr[j]),
        .s_wvalid(b_wvalid[j]), .s_wready(b_wready[j]), .s_wdata(b_wdata[j]),
        .s_wmask(b_wmask[j]), .s_rvalid(b_rvalid[j]), .s_rdata(b_rdata[j]),
        .s_wdone(b_wdone[j]), .s_err(b_err[j]),
        .m_cvalid(cv), .m_cready(cr), .m_cwe(cwe), .m_caddr(ca), .m_wvalid(wv), .m_wready(wr),
        .m_wdata(wd), .m_wmask(wm), .m_rvalid(rv), .m_rdata(rd), .m_wdone(wdone), .m_err(err));
      (* KEEP_HIERARCHY = "YES" *)
      otpu_native_reg u_rx (.clk, .rst(rst_c[j]),
        .s_cvalid(cv), .s_cready(cr), .s_cwe(cwe), .s_caddr(ca), .s_wvalid(wv), .s_wready(wr),
        .s_wdata(wd), .s_wmask(wm), .s_rvalid(rv), .s_rdata(rd), .s_wdone(wdone), .s_err(err),
        .m_cvalid(n_cvalid[j]), .m_cready(n_cready[j]), .m_cwe(n_cwe[j]), .m_caddr(n_caddr[j]),
        .m_wvalid(n_wvalid[j]), .m_wready(n_wready[j]), .m_wdata(n_wdata[j]),
        .m_wmask(n_wmask[j]), .m_rvalid(n_rvalid[j]), .m_rdata(n_rdata[j]),
        .m_wdone(n_wdone[j]), .m_err(n_err[j]));
    end else begin : g_wire
      assign n_cvalid[j] = b_cvalid[j]; assign b_cready[j] = n_cready[j];
      assign n_cwe[j] = b_cwe[j];       assign n_caddr[j] = b_caddr[j];
      assign n_wvalid[j] = b_wvalid[j]; assign b_wready[j] = n_wready[j];
      assign n_wdata[j] = b_wdata[j];   assign n_wmask[j] = b_wmask[j];
      assign b_rvalid[j] = n_rvalid[j]; assign b_rdata[j] = n_rdata[j];
      assign b_wdone[j] = n_wdone[j];   assign b_err[j] = n_err[j];
    end
    logic bid, rid;                   // (XIDW 1: XDMA's IDs stay in the SmartConnect)
    // ARD 256: read credits for the HBM path's round trip at 250 MHz (and the SLR crossing), with otpu_hbm_port's
    // RD 64 and otpu_board's NRD 256, as the FK33's (otpu_fk33_sys: decode +24% over 64 credits each)
    // BRAM: the read data's clock-crossing FIFO (u_ar, 256 x 520 bits) and the order FIFO in block RAM, not LUT
    // RAM: SLR0's CLBs were 93% used, 86k LUTs of LUT RAM (u_ar some 2,300 a channel), its block RAM 35% (f35d0004;
    // QUAD's f35d0009 missed timing there, docs/sqrl-fk35.md). (RQ, AREG: for a fast uclk, off at 250 MHz; the
    // 400 MHz builds did not close)
    otpu_mem_ch #(.XIDW(1), .ARD(256), .BRAM(1'b1)) u_ch (
      .clk, .rst(rst_c[j]),
      .n_cvalid(n_cvalid[j]), .n_cready(n_cready[j]), .n_cwe(n_cwe[j]), .n_caddr(n_caddr[j]),
      .n_wvalid(n_wvalid[j]), .n_wready(n_wready[j]), .n_wdata(n_wdata[j]),
      .n_wmask(n_wmask[j]), .n_rvalid(n_rvalid[j]), .n_rdata(n_rdata[j]),
      .n_wdone(n_wdone[j]), .n_err(n_err[j]),
      .xclk, .xrst,
      .x_awvalid(x_awvalid[j]), .x_awready(x_awready[j]), .x_awid(1'b0),
      .x_awaddr({1'b0, x_awaddr[j]}), .x_awlen(x_awlen[j]),
      .x_wvalid(x_wvalid[j]), .x_wready(x_wready[j]), .x_wdata(x_wdata[j]),
      .x_wstrb(x_wstrb[j]), .x_wlast(x_wlast[j]),
      .x_bvalid(x_bvalid[j]), .x_bready(x_bready[j]), .x_bid(bid), .x_bresp(x_bresp[j]),
      .x_arvalid(x_arvalid[j]), .x_arready(x_arready[j]), .x_arid(1'b0),
      .x_araddr({1'b0, x_araddr[j]}), .x_arlen(x_arlen[j]),
      .x_rvalid(x_rvalid[j]), .x_rready(x_rready[j]), .x_rid(rid), .x_rdata(x_rdata[j]),
      .x_rresp(x_rresp[j]), .x_rlast(x_rlast[j]),
      .uclk(uclk[j]), .urst(urst[j]),
      .c_cmd_valid(c_cmd_valid[j]), .c_cmd_ready(c_cmd_ready[j]), .c_cmd_we(c_cmd_we[j]),
      .c_cmd_addr(c_cmd_addr[j]),
      .c_wdata_valid(c_wdata_valid[j]), .c_wdata_ready(c_wdata_ready[j]),
      .c_wdata_data(c_wdata_data[j]), .c_wdata_we(c_wdata_we[j]),
      .c_rdata_valid(c_rdata_valid[j]), .c_rdata_data(c_rdata_data[j]));
  end

  // HAS_I2C 0: the board's I2C bus (through its mux) is wired to bus 0's pins, but CAPS bit2 stays
  // clear (the host's I2C tools expect the YPCB's LM73 and PCIe SMBus there)
  otpu_board #(.S(S), .D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .ACT_BLOCKS(ACT_BLOCKS),
               .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
               .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
               .MXU_CL(MXU_CL), .VPU_CL(VPU_CL), .ULANES(ULANES), .CORE_KHZ(CORE_KHZ),
               .BUILD_ID(BUILD_ID), .DDR_MTS(DDR_MTS), .TRACE_DEPTH(TRACE_DEPTH),
               .PQ_WIN(PQ_WIN), .HAS_I2C(1'b0), .CHASH(CHASH), .NRD(256), .DSTEP(DSTEP),
               .HOSTCAL(1'b0), .SEQ_WC(SEQ_WC), .PLATFORM(PLATFORM), .ARCH_HASH(ARCH_HASH),
               .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ),
               .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH), .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP),
               .TMEM_GST(TMEM_GST), .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM),
               .DUAL(DUAL), .QUAD(QUAD), .NQMEM(NQMEM), .MEMX(MEMX)) u_board (
    .clk, .clk_f, .rst(rst_b), .ctl_mrst, .calib, .ded, .temp, .led, .i2c_lo, .i2c_pin,
    .s_ctl_awaddr, .s_ctl_awvalid, .s_ctl_awready, .s_ctl_wdata, .s_ctl_wstrb, .s_ctl_wvalid,
    .s_ctl_wready, .s_ctl_bresp, .s_ctl_bvalid, .s_ctl_bready, .s_ctl_araddr, .s_ctl_arvalid,
    .s_ctl_arready, .s_ctl_rdata, .s_ctl_rresp, .s_ctl_rvalid, .s_ctl_rready,
    .n_cvalid(b_cvalid), .n_cready(b_cready), .n_cwe(b_cwe), .n_caddr(b_caddr),
    .n_wvalid(b_wvalid), .n_wready(b_wready), .n_wdata(b_wdata), .n_wmask(b_wmask),
    .n_rvalid(b_rvalid), .n_rdata(b_rdata), .n_wdone(b_wdone), .n_err(b_err));
endmodule
