// openTPU: S slices, each with its own DRAM, plus the collective unit (simulation top).
// AXI = 0: the behavioural fixed-latency DRAM. AXI = 1: the board's memory path -- the native
// adapter (otpu_native_dram) in front of a two-channel native memory model with random stalls
// and latency (sim/verilator/otpu_native_mem.sv; requires D = 128); the dump prints the
// adapter's counters. AXI = 2: the same adapter in front of the card's channels as they are
// (sim/verilator/otpu_ldc_mem.sv: per channel the bridge otpu_mem_ch and LiteDRAM's own
// controller, generated with the production core's settings, in the controller clock).
// A slice of more than 4 GiB (DAW > 32, otpu_pkg): the ports' word addresses are AWW bits, the
// native adapter's channel beats BAW bits (the slice's 2^DAW bytes on two channels, DAW - 7, or on
// four with DUAL, DAW - 8; 25 at least), and the memory models hold a window of DRAM_WORDS words
// per 4 GiB region (otpu_dram, otpu_native_mem NREG: images and dumps of NREG windows, in region
// order); the LiteDRAM path (AXI 2) is DAW 32's only.
module otpu_top
  import otpu_pkg::*;
#(
  parameter int S          = 1,
  parameter int D          = 32,
  parameter int MCOLS      = 8,
  parameter int ACT_BLOCKS = 64,
  parameter int ACT_ROWS   = MCOLS,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 16,
  parameter int DRAM_WORDS = 1 << 18,
  parameter int DRAM_LAT   = 8,
  parameter int FIFO_DEPTH = 128,
  parameter int LANES      = 8,
  parameter int WIN        = 32,
  parameter int RPB        = 4,
  parameter int WPB        = 2,
  parameter int MXU_IMPL   = 2,
  parameter int MXU_CL     = 16,
  parameter int MXU_E      = 0,       // MXU epilogue lanes (0: MCOLS / 2; otpu_mxu E)
  parameter int QUANT_XD   = 1024,    // quantizer row buffer chunks (0: two passes over TMEM; otpu_quant XD)
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,
  parameter int ULANES     = LANES,   // TMEM lanes of the MXU and the quantizer
  parameter int AXI        = 0,       // memory path: see the top
  parameter int TMEM_IMPL  = 0,       // TMEM: 0 a copy per read port, 1 banked (otpu_slice)
  parameter bit TMEM_RQ    = 1'b0,    // TMEM_IMPL 1: request-priority bank addresses (otpu_slice)
  parameter int TMEM_NCP   = 3,       // TMEM_IMPL 1: copies (otpu_slice)
  parameter int TMEM_NSB   = 8,       // TMEM_IMPL 3: sub-banks per bank (otpu_slice)
  parameter int TMEM_SBP   = 3,       // TMEM_IMPL 3: the sub-bank field's word-address bit
  parameter int TMEM_SBH   = 0,       // TMEM_IMPL 3: the sub-bank hashed (otpu_slice)
  parameter bit TMEM_RPL   = 1'b0,    // TMEM_IMPL 3: lost reads replayed (otpu_slice)
  parameter bit TMEM_TDP   = 1'b0,    // TMEM_IMPL 3: true dual-port RAMs (otpu_slice)
  parameter bit TMEM_GST   = 1'b0,    // TMEM_IMPL 3: QUANT 3 and COLL the DMA's guests (otpu_slice)
  parameter int TMEM_PMP   = 0,       // TMEM_TDP: 2 the pumped RAMs' model (otpu_slice)
  parameter int TMEM_PRK   = 0,       // TMEM_IMPL 3: the read ports' ranks (otpu_slice)
  parameter int TMEM_SDM   = -1,      // TMEM_TDP: the read ports' RAM sides (otpu_slice)
  parameter bit SEQ_WC     = 1'b0,    // EXPERIMENTAL: otpu_seq's wait counts (otpu_slice)
  parameter bit COLL_GATHER = 1'b1,   // 0 (S = 1 only): the collective unit runs BAR alone
                                      // (otpu_coll_bar), as the card builds it (otpu_board)
  parameter bit DUAL       = 1'b0     // port B chunk pairs: int8 MM PAIR (otpu_slice, otpu_mxu); the
                                      // board's path (AXI 1) on four channels (otpu_native_dram2)
) (
  input  logic          clk,
  input  logic          clk_f,         // TMEM_PMP 1: 2 x clk, rising with it (otpu_slice)
  input  logic          sys_rst,
  input  logic          rst,
  input  logic [31:0]   rinit [8],     // the run's arguments (R8..R15 at the start)
  input  logic          ld_start,
  input  logic [31:0]   ld_addr,
  input  logic [31:0]   ld_n,
  output logic          ld_busy,
  output logic          all_halted,
  output logic          any_error,
  output logic [31:0]   icount [S],
  input  logic          dump
);
  localparam int NREG = 1 << RGN;                   // DRAM windows (4 GiB regions) of the models
  localparam int BAW1 = (DAW - 7 > 25) ? DAW - 7 : 25;   // channel beat bits: two channels
  localparam int BAW2 = (DAW - 8 > 25) ? DAW - 8 : 25;   // ... four (DUAL)
  logic [S-1:0] halted, error, coll_req, coll_gl, ldb;
  assign ld_busy = |ldb;
  cmd_t         coll_cmd [S];
  logic         coll_ack;
  logic [S-1:0][LANES-1:0]       coll_ren;
  logic [S-1:0][LANES-1:0][31:0] coll_raddr, coll_rdata;
  logic [LANES-1:0]              coll_wen;
  logic [LANES-1:0][31:0]        coll_waddr, coll_wdata;
  wire coll_gnt = &coll_gl;

  for (genvar s = 0; s < S; s++) begin : g_slice
    logic          a_req, a_we, a_rvalid, b_req, b_tag, b_we, b_par, b_rvalid, b_rtag;
    logic [D/4-1:0] b_wmask;
    logic [D*8-1:0] b_wdata;
    logic [AWW-1:0] a_addr, b_addr, sw_addr;
    logic [31:0]   a_wdata;
    logic [255:0]  a_rdata;
    logic [3:0]    a_be;
    logic [D*8-1:0] b_rdata, b_rdata2;
    logic          b_two;
    logic          a_rdy, b_rdy, wr_idle, rd_idle, sw_req, sw_rdy, a_inval;
    logic [31:0]   sw_wdata;
    logic [3:0]    sw_be;

    if (AXI == 0) begin : g_dram
      assign a_rdy = 1'b1;
      assign sw_rdy = 1'b1;
      assign b_rdy = 1'b1;
      assign wr_idle = 1'b1;
      otpu_dram #(.WORDS(DRAM_WORDS), .D(D), .LAT(DRAM_LAT), .SID(s), .AW(AWW), .NREG(NREG)) u_dram (
        .clk, .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
        .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_two, .b_rvalid, .b_rtag, .b_rdata,
        .b_rdata2, .rd_idle, .dump);
    end else if (DUAL) begin : g_native2
      // four channels: otpu_native_dram2's two adapters, the model's NC 4 map
      initial if (AXI == 2) $fatal(1, "otpu_top: DUAL has no LiteDRAM path (AXI 2)");
      logic [3:0] cvalid, cready, cwe, wvalid, wready, rvalid;
      logic [3:0][BAW2-1:0] caddr;
      logic [3:0][511:0] wdata, rdata;
      logic [3:0][63:0] wmask;
      logic [3:0][15:0] wdone;
      otpu_native_dram2 #(.D(D), .AW(AWW), .BAW(BAW2)) u_adapt (
        .clk, .rst(sys_rst), .a_flush(rst || ld_start || a_inval),   // as otpu_board
        .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
        .a_be_x(a_be), .a_rvalid, .a_rdata,
        .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_two, .b_rvalid,
        .b_rtag, .b_rdata, .b_rdata2, .wr_idle, .rd_idle,
        .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
        .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
        .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone));
      otpu_native_mem #(.WORDS(DRAM_WORDS), .LAT(DRAM_LAT), .SID(s), .NC(4), .BAW(BAW2),
                        .NREG(NREG)) u_mem (
        .clk, .rst(sys_rst), .run_rst(rst),
        .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
        .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
        .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone), .dump);
    end else begin : g_native
      assign b_rdata2 = '0;
      logic [1:0] cvalid, cready, cwe, wvalid, wready, rvalid;
      logic [1:0][BAW1-1:0] caddr;
      logic [1:0][511:0] wdata, rdata;
      logic [1:0][63:0] wmask;
      logic [1:0][15:0] wdone;
      otpu_native_dram #(.D(D), .AW(AWW), .BAW(BAW1)) u_adapt (
        .clk, .rst(sys_rst), .a_flush(rst || ld_start || a_inval),   // as otpu_board
        .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
        .a_be_x(a_be), .a_rvalid, .a_rdata,
        .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag,
        .b_rdata, .b_ravail(), .b_rrdy(1'b1), .a_ravail(), .a_rrdy(1'b1), .wr_idle, .rd_idle,
        .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
        .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
        .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone));
      initial if (AXI == 2 && WIDE) $fatal(1, "otpu_top: the LiteDRAM path (AXI 2) is DAW 32's");
      if (AXI == 2) begin : g_ldc
        otpu_ldc_mem #(.WORDS(DRAM_WORDS), .SID(s)) u_mem (
          .clk, .rst(sys_rst),
          .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
          .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
          .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone), .dump);
      end else begin : g_model
        otpu_native_mem #(.WORDS(DRAM_WORDS), .LAT(DRAM_LAT), .SID(s), .BAW(BAW1), .NREG(NREG)) u_mem (
          .clk, .rst(sys_rst), .run_rst(rst),
          .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
          .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
          .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone), .dump);
      end
      // the adapter's counters (rtlsim: A runs and fill reads, the partial writes by source)
      always @(posedge clk) if (dump)
        for (int c = 0; c < 2; c++) begin
          $write("NATIVE ch%0d a_runs=%0d a_rd=%0d b_rd=%0d sw_rd=%0d ", c, u_adapt.st_arun[c],
                 u_adapt.st_ard[c], u_adapt.st_brd[c], u_adapt.st_srd[c]);
          $display("b_wr=%0d a_wr=%0d sw_wr=%0d part_b=%0d part_a=%0d part_sw=%0d",
                   u_adapt.st_bwr[c], u_adapt.st_awr[c], u_adapt.st_swr[c], u_adapt.st_pb[c],
                   u_adapt.st_pa[c], u_adapt.st_ps[c]);
        end
    end

    otpu_slice #(.SID(s), .S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .ACT_ROWS(ACT_ROWS),
                 .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS),
                 .FIFO_DEPTH(FIFO_DEPTH), .LANES(LANES), .WIN(WIN), .RPB(RPB),
                 .WPB(WPB), .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .MXU_E(MXU_E), .QUANT_XD(QUANT_XD), .VPU_CL(VPU_CL), .ULANES(ULANES),
                 .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ), .TMEM_NCP(TMEM_NCP),
                 .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH),
                 .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST),
                 .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM), .SEQ_WC(SEQ_WC),
                 .DUAL(DUAL))
    u_slice (
      .clk, .clk_f, .sys_rst, .rst, .rinit, .ld_start, .ld_addr, .ld_n, .ld_busy(ldb[s]),
      .a_rdy, .b_rdy, .sw_rdy, .wr_idle, .rd_idle,
      .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
      .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_two, .b_rvalid, .b_rtag,
      .b_rdata, .b_rdata2,
      .coll_req(coll_req[s]), .coll_cmd(coll_cmd[s]), .coll_ack,
      .coll_ren(coll_ren[s]), .coll_raddr(coll_raddr[s]), .coll_rdata(coll_rdata[s]),
      .coll_wen, .coll_waddr, .coll_wdata, .coll_gnt_local(coll_gl[s]), .coll_gnt,
      .halted(halted[s]), .error(error[s]), .wait_to(), .a_inval, .icount(icount[s]), .pf(), .dump);
  end

  logic coll_gerr;                    // a GATHER on the BAR-only collective unit
  initial if (S > 1 && !COLL_GATHER) $fatal(1, "otpu_top: COLL_GATHER 0 needs S = 1");
  if (COLL_GATHER) begin : g_coll
    otpu_coll #(.S(S), .LANES(LANES)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata));
    assign coll_gerr = 1'b0;
  end else begin : g_coll_bar
    otpu_coll_bar #(.S(S), .LANES(LANES)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata), .gerr(coll_gerr));
  end

  assign all_halted = &halted;
  assign any_error  = |error || coll_gerr;
endmodule
