// The accelerator and XDMA's DMA on the SQRL FK33's HBM (otpu_fpga_top_fk33: one otpu_hbm_port per channel on
// the HBM IP's AXI ports): otpu_board with S = 1 slice (DUAL: on board channels 0 .. 3, otpu_native_dram2; else 0
// and 1, otpu_native_dram) and one otpu_mem_ch per channel in front of that channel's controller ports c_* in uclk
// (the adapters' and the HBM AXI ports' clock), meeting XDMA's master for that channel there (the block design's
// SmartConnect sc_dma splits XDMA's DMA by card address bits 32:31, channel j at j * 2 GiB, and converts it to 128
// bits). A copy of the BCU1525's otpu_bcu_sys for this board: one SLR, so no SLR crossing on the native ports (the
// BCU1525's SLL).
//   calib: each channel's calibration flag in its controller's clock (STATUS CALIB0 / CALIB1: the slice's first /
//        second channel of each pair; synchronized in otpu_board). ded: each channel's ECC has counted an uncorrectable word
//        (STATUS ECC_DED; the same).
// The parameters are the target's (targets/sqrl-fk33.toml through opentpu.target rtl_params; the defaults are its
// values, tests/test_target.py).
module otpu_fk33_sys #(
  parameter int S          = 1,
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
  parameter int PLATFORM   = 5,
  parameter logic [31:0] ARCH_HASH = 32'h0,
  parameter bit DUAL       = 1'b1,          // S = 1 on four channels, port B pairs (otpu_board DUAL)
  parameter int NC         = (DUAL ? 4 : 2) * S,  // memory channels (two per slice; DUAL four)
  // the channels' beat address bits (otpu_board's: 25, 2 GiB a channel, up to DAW 33 with DUAL; a 16 GB part at
  // DAW 34: 26, 4 GiB channels), and so XDMA's address bits of a channel, BAW + 6
  parameter int BAW        = (otpu_pkg::DAW - $clog2(NC / S) - 6 > 25) ? otpu_pkg::DAW - $clog2(NC / S) - 6 : 25
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
  // bursts of full-width beats; the channel's 2^(BAW + 6) bytes (2 GiB) in address bits BAW+5:0
  input  logic [NC-1:0]           x_awvalid,
  output logic [NC-1:0]           x_awready,
  input  logic [NC-1:0][BAW+5:0]  x_awaddr,
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
  input  logic [NC-1:0][BAW+5:0]  x_araddr,
  input  logic [NC-1:0][7:0]      x_arlen,
  output logic [NC-1:0]           x_rvalid,
  input  logic [NC-1:0]           x_rready,
  output logic [NC-1:0][127:0]    x_rdata,
  output logic [NC-1:0][1:0]      x_rresp,
  output logic [NC-1:0]           x_rlast,
  // the channels' controller ports, channel j in uclk[j], two ports each ([channel][port];
  // otpu_mem_ch's c_*: port p takes the banks of parity p; the beat within the 2 GiB a slice
  // addresses, BAW bits: otpu_hbm_port's c_cmd_addr)
  input  logic [NC-1:0]           uclk,
  input  logic [NC-1:0]           urst,
  output logic [NC-1:0][1:0]      c_cmd_valid,
  input  logic [NC-1:0][1:0]      c_cmd_ready,
  output logic [NC-1:0][1:0]      c_cmd_we,
  output logic [NC-1:0][1:0][BAW-1:0] c_cmd_addr,
  output logic [NC-1:0][1:0]      c_wdata_valid,
  input  logic [NC-1:0][1:0]      c_wdata_ready,
  output logic [NC-1:0][1:0][511:0] c_wdata_data,
  output logic [NC-1:0][1:0][63:0] c_wdata_we,
  input  logic [NC-1:0][1:0]      c_rdata_valid,
  input  logic [NC-1:0][1:0][511:0] c_rdata_data
);
  // the accelerator's native masters, per channel (otpu_board's n_* to otpu_mem_ch's)
  logic [NC-1:0]        n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid, n_err;
  logic [NC-1:0][BAW-1:0] n_caddr;
  localparam int XAW = (BAW + 6 > 32) ? BAW + 6 : 32;   // otpu_mem_ch's
  logic [NC-1:0][511:0] n_wdata, n_rdata;
  logic [NC-1:0][63:0]  n_wmask;
  logic [NC-1:0][15:0]  n_wdone;

  for (genvar j = 0; j < NC; j++) begin : g_ch
    logic bid, rid;                   // (XIDW 1: XDMA's IDs stay in the SmartConnect)
    // ARD 256: read credits for the HBM path's round trip at 250 MHz, with otpu_hbm_port's RD 64 (4 x 64 read
    // slots) and otpu_board's NRD 256 (otpu_native_dram): 64 credits each left decode at ~0.78 of the
    // 2 x 64 B/cycle port, 256 at ~0.96 (Qwen3.5-0.8B int8 30.8 -> 38.1 tok/s, 2B 12.9 -> 16.0, 4B mix 8.5 -> 10.5)
    otpu_mem_ch #(.XIDW(1), .ARD(256), .BAW(BAW)) u_ch (
      .clk, .rst,
      .n_cvalid(n_cvalid[j]), .n_cready(n_cready[j]), .n_cwe(n_cwe[j]), .n_caddr(n_caddr[j]),
      .n_wvalid(n_wvalid[j]), .n_wready(n_wready[j]), .n_wdata(n_wdata[j]),
      .n_wmask(n_wmask[j]), .n_rvalid(n_rvalid[j]), .n_rdata(n_rdata[j]),
      .n_wdone(n_wdone[j]), .n_err(n_err[j]),
      .xclk, .xrst,
      .x_awvalid(x_awvalid[j]), .x_awready(x_awready[j]), .x_awid(1'b0),
      .x_awaddr(XAW'(x_awaddr[j])), .x_awlen(x_awlen[j]),
      .x_wvalid(x_wvalid[j]), .x_wready(x_wready[j]), .x_wdata(x_wdata[j]),
      .x_wstrb(x_wstrb[j]), .x_wlast(x_wlast[j]),
      .x_bvalid(x_bvalid[j]), .x_bready(x_bready[j]), .x_bid(bid), .x_bresp(x_bresp[j]),
      .x_arvalid(x_arvalid[j]), .x_arready(x_arready[j]), .x_arid(1'b0),
      .x_araddr(XAW'(x_araddr[j])), .x_arlen(x_arlen[j]),
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
               .DUAL(DUAL), .BAW(BAW)) u_board (
    .clk, .clk_f, .rst, .ctl_mrst, .calib, .ded, .temp, .led, .i2c_lo, .i2c_pin,
    .s_ctl_awaddr, .s_ctl_awvalid, .s_ctl_awready, .s_ctl_wdata, .s_ctl_wstrb, .s_ctl_wvalid,
    .s_ctl_wready, .s_ctl_bresp, .s_ctl_bvalid, .s_ctl_bready, .s_ctl_araddr, .s_ctl_arvalid,
    .s_ctl_arready, .s_ctl_rdata, .s_ctl_rresp, .s_ctl_rvalid, .s_ctl_rready,
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask,
    .n_rvalid, .n_rdata, .n_wdone, .n_err);
endmodule
