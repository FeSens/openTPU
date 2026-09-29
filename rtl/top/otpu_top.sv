// openTPU: S slices, each with its own DRAM, plus the collective unit (simulation top).
// AXI = 0: the behavioural fixed-latency DRAM. AXI = 1: the board's memory path -- the AXI
// adapter (otpu_axi_dram) in front of a two-channel AXI memory model with random stalls and
// latency (sim/verilator/otpu_axi_mem.sv; requires D = 128). AXI = 2: the native memory path --
// the native adapter (otpu_native_dram) in front of a two-channel native memory model
// (sim/verilator/otpu_native_mem.sv; requires D = 128); the dump prints the adapter's counters.
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
  parameter int MXU_IMPL   = 0,
  parameter int MXU_CL     = 16,
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,
  parameter int ULANES     = LANES,   // TMEM lanes of the MXU and the quantizer
  parameter int AXI        = 0,       // memory path: see the top
  parameter int AXI_BL     = 8,       // AXI adapter: port B read burst, beats (timing only)
  parameter int AXI_WBL    = 8        // AXI adapter: port B write burst, beats (timing only)
) (
  input  logic          clk,
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
    logic          a_req, a_we, a_rvalid, b_req, b_tag, b_we, b_rvalid, b_rtag;
    logic [D/4-1:0] b_wmask;
    logic [D*8-1:0] b_wdata;
    logic [31:0]   a_addr, a_wdata, a_rdata, a_rdata2, b_addr;
    logic [3:0]    a_be;
    logic [D*8-1:0] b_rdata;
    logic          a_rdy, b_rdy, wr_idle, sw_req, sw_rdy;
    logic [31:0]   sw_addr, sw_wdata;
    logic [3:0]    sw_be;

    if (AXI == 0) begin : g_dram
      assign a_rdy = 1'b1;
      assign sw_rdy = 1'b1;
      assign b_rdy = 1'b1;
      assign wr_idle = 1'b1;
      otpu_dram #(.WORDS(DRAM_WORDS), .D(D), .LAT(DRAM_LAT), .SID(s)) u_dram (
        .clk, .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata, .a_rdata2,
        .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_rvalid, .b_rtag, .b_rdata, .dump);
    end else if (AXI == 2) begin : g_native
      logic [1:0] cvalid, cready, cwe, wvalid, wready, rvalid;
      logic [1:0][24:0] caddr;
      logic [1:0][511:0] wdata, rdata;
      logic [1:0][63:0] wmask;
      logic [1:0][15:0] wdone;
      otpu_native_dram #(.D(D)) u_adapt (
        .clk, .rst(sys_rst),
        .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
        .a_be_x(a_be), .a_rvalid, .a_rdata, .a_rdata2,
        .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_rvalid, .b_rtag, .b_rdata,
        .wr_idle,
        .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
        .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
        .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone));
      otpu_native_mem #(.WORDS(DRAM_WORDS), .LAT(DRAM_LAT), .SID(s)) u_mem (
        .clk, .rst(sys_rst),
        .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
        .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
        .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone), .dump);
      // the adapter's counters (rtlsim: A runs and fill reads, the partial writes by source)
      always @(posedge clk) if (dump)
        for (int c = 0; c < 2; c++) begin
          $write("NATIVE ch%0d a_runs=%0d a_rd=%0d b_rd=%0d sw_rd=%0d ", c, u_adapt.st_arun[c],
                 u_adapt.st_ard[c], u_adapt.st_brd[c], u_adapt.st_srd[c]);
          $display("b_wr=%0d a_wr=%0d sw_wr=%0d part_b=%0d part_a=%0d part_sw=%0d",
                   u_adapt.st_bwr[c], u_adapt.st_awr[c], u_adapt.st_swr[c], u_adapt.st_pb[c],
                   u_adapt.st_pa[c], u_adapt.st_ps[c]);
        end
    end else begin : g_axi
      logic [1:0] awvalid, awready, awid, wvalid, wready, bvalid, bready, bid;
      logic [1:0] arvalid, arready, arid, rvalid, rready, rid, rlast;
      logic [1:0][31:0] awaddr, araddr;
      logic [1:0][7:0] arlen, awlen;
      logic [1:0] wlast;
      logic [1:0][511:0] wdata, rdata;
      logic [1:0][63:0] wstrb;
      logic [1:0][1:0] bresp, rresp;
      logic axi_err;
      otpu_axi_dram #(.D(D), .BL(AXI_BL), .WBL(AXI_WBL)) u_adapt (
        .clk, .rst(sys_rst),
        .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
        .a_be_x(a_be), .a_rvalid, .a_rdata, .a_rdata2,
        .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_rvalid, .b_rtag, .b_rdata,
        .wr_idle,
        .m_awvalid(awvalid), .m_awready(awready), .m_awaddr(awaddr), .m_awid(awid),
        .m_awlen(awlen), .m_wvalid(wvalid), .m_wready(wready), .m_wdata(wdata), .m_wstrb(wstrb),
        .m_wlast(wlast),
        .m_bvalid(bvalid), .m_bready(bready), .m_bid(bid), .m_bresp(bresp),
        .m_arvalid(arvalid), .m_arready(arready), .m_araddr(araddr), .m_arlen(arlen), .m_arid(arid),
        .m_rvalid(rvalid), .m_rready(rready), .m_rid(rid), .m_rdata(rdata), .m_rresp(rresp),
        .m_rlast(rlast), .err(axi_err));
      otpu_axi_mem #(.WORDS(DRAM_WORDS), .LAT(DRAM_LAT), .SID(s)) u_mem (
        .clk, .rst(sys_rst),
        .s_awvalid(awvalid), .s_awready(awready), .s_awaddr(awaddr), .s_awid(awid),
        .s_awlen(awlen), .s_wvalid(wvalid), .s_wready(wready), .s_wdata(wdata), .s_wstrb(wstrb),
        .s_wlast(wlast),
        .s_bvalid(bvalid), .s_bready(bready), .s_bid(bid), .s_bresp(bresp),
        .s_arvalid(arvalid), .s_arready(arready), .s_araddr(araddr), .s_arlen(arlen), .s_arid(arid),
        .s_rvalid(rvalid), .s_rready(rready), .s_rid(rid), .s_rdata(rdata), .s_rresp(rresp),
        .s_rlast(rlast), .dump);
    end

    otpu_slice #(.SID(s), .S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .ACT_ROWS(ACT_ROWS),
                 .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS),
                 .FIFO_DEPTH(FIFO_DEPTH), .LANES(LANES), .WIN(WIN), .RPB(RPB),
                 .WPB(WPB), .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .VPU_CL(VPU_CL), .ULANES(ULANES))
    u_slice (
      .clk, .sys_rst, .rst, .rinit, .ld_start, .ld_addr, .ld_n, .ld_busy(ldb[s]),
      .a_rdy, .b_rdy, .sw_rdy, .wr_idle,
      .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata, .a_rdata2,
      .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_rvalid, .b_rtag, .b_rdata,
      .coll_req(coll_req[s]), .coll_cmd(coll_cmd[s]), .coll_ack,
      .coll_ren(coll_ren[s]), .coll_raddr(coll_raddr[s]), .coll_rdata(coll_rdata[s]),
      .coll_wen, .coll_waddr, .coll_wdata, .coll_gnt_local(coll_gl[s]), .coll_gnt,
      .halted(halted[s]), .error(error[s]), .icount(icount[s]), .pf(), .dump);
  end

  otpu_coll #(.S(S), .LANES(LANES)) u_coll (
    .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gnt), .ack(coll_ack),
    .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
    .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata));

  assign all_halted = &halted;
  assign any_error  = |error;
endmodule
