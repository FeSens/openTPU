// The accelerator as built for the YPCB-00338 board: one slice, the DRAM adapter onto the two
// DDR3 channels (AXI4 masters m0/m1, 512-bit, single beats) and the host control registers
// (AXI4-Lite slave). Everything runs on the core clock; the block design's interconnect does
// the clock and width conversion to the memory controllers and the PCIe bridge. The control
// block also holds the free-running activity counters and reads out the hardware trace
// (otpu_trace; docs/observability.md).
module otpu_board #(
  parameter int D          = 128,
  parameter int MCOLS      = 2,
  parameter int ACT_BLOCKS = 128,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 15,
  parameter int FIFO_DEPTH = 1024,
  parameter int LANES      = 8,
  parameter int WIN        = 16,
  parameter int RPB        = 8 * LANES,   // every read lane (TMEM is replicated per read port):
                                          // the slice's shallow write-mask arbiter
  parameter int WPB        = 1,
  parameter int MXU_IMPL   = 0,
  parameter int MXU_CL     = 16,
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,  // VPU lanes with exp2/recip/rsqrt
  parameter int ULANES     = LANES,   // TMEM lanes of the MXU and the quantizer
  parameter logic [31:0] BASE0 = 32'h0000_0000,
  parameter logic [31:0] BASE1 = 32'h8000_0000,
  parameter int CORE_KHZ    = 100000,  // the core clock (CORE_KHZ register)
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int TRACE_DEPTH = 16384,   // trace records (a power of two; 0: no trace buffer)
  parameter int TRACE_QD    = 32,      // trace capture queue (cycles with events)
  parameter int PQ_WIN      = 1024     // cycles per P/Q counter window
) (
  input  logic         clk,
  input  logic         rst,            // synchronous, active high
  input  logic [1:0]   calib,          // memory controllers calibrated (any clock domain)
  input  logic [11:0]  temp,           // XADC die-temperature code (any clock domain, slow)
  output logic [2:0]   led,
  // ---- control: AXI4-Lite slave
  input  logic [11:0]  s_ctl_awaddr,
  input  logic         s_ctl_awvalid,
  output logic         s_ctl_awready,
  input  logic [31:0]  s_ctl_wdata,
  input  logic [3:0]   s_ctl_wstrb,
  input  logic         s_ctl_wvalid,
  output logic         s_ctl_wready,
  output logic [1:0]   s_ctl_bresp,
  output logic         s_ctl_bvalid,
  input  logic         s_ctl_bready,
  input  logic [11:0]  s_ctl_araddr,
  input  logic         s_ctl_arvalid,
  output logic         s_ctl_arready,
  output logic [31:0]  s_ctl_rdata,
  output logic [1:0]   s_ctl_rresp,
  output logic         s_ctl_rvalid,
  input  logic         s_ctl_rready,
  // ---- memory channel 0: AXI4 master
  output logic [0:0]   m0_axi_awid,
  output logic [31:0]  m0_axi_awaddr,
  output logic [7:0]   m0_axi_awlen,
  output logic [2:0]   m0_axi_awsize,
  output logic [1:0]   m0_axi_awburst,
  output logic         m0_axi_awlock,
  output logic [3:0]   m0_axi_awcache,
  output logic [2:0]   m0_axi_awprot,
  output logic [3:0]   m0_axi_awqos,
  output logic         m0_axi_awvalid,
  input  logic         m0_axi_awready,
  output logic [511:0] m0_axi_wdata,
  output logic [63:0]  m0_axi_wstrb,
  output logic         m0_axi_wlast,
  output logic         m0_axi_wvalid,
  input  logic         m0_axi_wready,
  input  logic [0:0]   m0_axi_bid,
  input  logic [1:0]   m0_axi_bresp,
  input  logic         m0_axi_bvalid,
  output logic         m0_axi_bready,
  output logic [0:0]   m0_axi_arid,
  output logic [31:0]  m0_axi_araddr,
  output logic [7:0]   m0_axi_arlen,
  output logic [2:0]   m0_axi_arsize,
  output logic [1:0]   m0_axi_arburst,
  output logic         m0_axi_arlock,
  output logic [3:0]   m0_axi_arcache,
  output logic [2:0]   m0_axi_arprot,
  output logic [3:0]   m0_axi_arqos,
  output logic         m0_axi_arvalid,
  input  logic         m0_axi_arready,
  input  logic [0:0]   m0_axi_rid,
  input  logic [511:0] m0_axi_rdata,
  input  logic [1:0]   m0_axi_rresp,
  input  logic         m0_axi_rlast,
  input  logic         m0_axi_rvalid,
  output logic         m0_axi_rready,
  // ---- memory channel 1: AXI4 master
  output logic [0:0]   m1_axi_awid,
  output logic [31:0]  m1_axi_awaddr,
  output logic [7:0]   m1_axi_awlen,
  output logic [2:0]   m1_axi_awsize,
  output logic [1:0]   m1_axi_awburst,
  output logic         m1_axi_awlock,
  output logic [3:0]   m1_axi_awcache,
  output logic [2:0]   m1_axi_awprot,
  output logic [3:0]   m1_axi_awqos,
  output logic         m1_axi_awvalid,
  input  logic         m1_axi_awready,
  output logic [511:0] m1_axi_wdata,
  output logic [63:0]  m1_axi_wstrb,
  output logic         m1_axi_wlast,
  output logic         m1_axi_wvalid,
  input  logic         m1_axi_wready,
  input  logic [0:0]   m1_axi_bid,
  input  logic [1:0]   m1_axi_bresp,
  input  logic         m1_axi_bvalid,
  output logic         m1_axi_bready,
  output logic [0:0]   m1_axi_arid,
  output logic [31:0]  m1_axi_araddr,
  output logic [7:0]   m1_axi_arlen,
  output logic [2:0]   m1_axi_arsize,
  output logic [1:0]   m1_axi_arburst,
  output logic         m1_axi_arlock,
  output logic [3:0]   m1_axi_arcache,
  output logic [2:0]   m1_axi_arprot,
  output logic [3:0]   m1_axi_arqos,
  output logic         m1_axi_arvalid,
  input  logic         m1_axi_arready,
  input  logic [0:0]   m1_axi_rid,
  input  logic [511:0] m1_axi_rdata,
  input  logic [1:0]   m1_axi_rresp,
  input  logic         m1_axi_rlast,
  input  logic         m1_axi_rvalid,
  output logic         m1_axi_rready
);
  import otpu_pkg::*;

  // ---- calibration flags from the memory controllers' clock domains
  (* ASYNC_REG = "TRUE" *) logic [1:0] cal_s1, cal_s2;
  always_ff @(posedge clk) begin
    cal_s1 <= calib;
    cal_s2 <= cal_s1;
  end

  // ---- die temperature: two flip-flops per bit, then a code is taken only when two
  // consecutive samples agree (it changes slowly: a sample caught mid-change is skipped). Valid
  // once channel 0 (which owns the XADC) is calibrated and has reported a reading.
  (* ASYNC_REG = "TRUE" *) logic [11:0] tmp_s1, tmp_s2;
  logic [11:0] tmp_s3, temp_q;
  logic        temp_v;
  always_ff @(posedge clk) begin
    tmp_s1 <= temp;
    tmp_s2 <= tmp_s1;
    tmp_s3 <= tmp_s2;
    if (rst) begin
      temp_q <= '0;
      temp_v <= 1'b0;
    end else if (tmp_s2 == tmp_s3 && cal_s2[0] && tmp_s3 != '0) begin
      temp_q <= tmp_s3;
      temp_v <= 1'b1;
    end
  end

  // ---- control
  logic run, ld_start, ld_busy, halted, error, wr_idle, axi_err;
  logic [31:0] ld_addr, ld_n, icount;
  logic a_req, a_we, a_rvalid, a_rdy, b_req, b_tag, b_we, b_rvalid, b_rtag, b_rdy;
  logic [31:0] a_addr, a_wdata, a_rdata, b_addr;
  logic [3:0]  a_be, sw_be;
  logic        sw_req, sw_rdy;
  logic [31:0] sw_addr, sw_wdata;
  logic [D/4-1:0] b_wmask;
  logic [D*8-1:0] b_wdata, b_rdata;

  perf_t pf;
  logic        tr_en, tr_stop, tr_clear, tr_busy;
  logic [31:0] tr_addr, tr_count, tr_drop;
  logic [63:0] tr_rdata;
  logic [1:0]  awvalid, awready, awid, wvalid, wready, bvalid, bready, bid;
  logic [1:0]  arvalid, arready, arid, rvalid, rready, rid, rlast;

  otpu_ctrl #(.D(D), .MCOLS(MCOLS), .LANES(LANES), .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID),
              .TRACE_DEPTH(TRACE_DEPTH), .PQ_WIN(PQ_WIN), .HAS_TEMP(1'b1)) u_ctrl (
    .clk, .rst,
    .s_awaddr(s_ctl_awaddr), .s_awvalid(s_ctl_awvalid), .s_awready(s_ctl_awready),
    .s_wdata(s_ctl_wdata), .s_wstrb(s_ctl_wstrb), .s_wvalid(s_ctl_wvalid),
    .s_wready(s_ctl_wready), .s_bresp(s_ctl_bresp), .s_bvalid(s_ctl_bvalid),
    .s_bready(s_ctl_bready), .s_araddr(s_ctl_araddr), .s_arvalid(s_ctl_arvalid),
    .s_arready(s_ctl_arready), .s_rdata(s_ctl_rdata), .s_rresp(s_ctl_rresp),
    .s_rvalid(s_ctl_rvalid), .s_rready(s_ctl_rready),
    .run, .ld_start, .ld_addr, .ld_n, .ld_busy, .halted, .error, .icount, .wr_idle, .axi_err,
    .calib(cal_s2),
    .b_rd(b_req && b_rdy && !b_we), .b_wr(b_req && b_rdy && b_we),
    .a_rd(a_req && a_rdy && !a_we), .a_wr(sw_req && sw_rdy), .b_wait(b_req && !b_rdy),
    .temp_v, .temp(temp_q),
    .mxu_busy(pf.sq.busy[U_MXU]), .mxu_mac(pf.mac), .vpu_busy(pf.sq.busy[U_VPU]),
    .qnt_busy(pf.sq.busy[U_Q]), .dma_busy(pf.sq.busy[U_DMA]), .tmem_deny(pf.deny),
    .dram_rd(2'(rvalid[0] && rready[0]) + 2'(rvalid[1] && rready[1])),
    .dram_wr(2'(wvalid[0] && wready[0]) + 2'(wvalid[1] && wready[1])),
    .dram_wait((b_req && !b_rdy) || (a_req && !a_rdy) || (sw_req && !sw_rdy)),
    .instr(pf.sq.ret),
    .tr_en, .tr_stop, .tr_clear, .tr_addr, .tr_count, .tr_drop, .tr_busy, .tr_rdata);

  // ---- hardware trace
  if (TRACE_DEPTH != 0) begin : g_trace
    otpu_trace #(.DEPTH(TRACE_DEPTH), .QD(TRACE_QD), .WIN(WIN)) u_trace (
      .clk, .rst, .pf, .en(tr_en && run), .stop(tr_stop), .clear(tr_clear), .raddr(tr_addr),
      .rdata(tr_rdata), .count(tr_count), .drop(tr_drop), .busy(tr_busy));
  end else begin : g_no_trace
    assign tr_rdata = '0;
    assign tr_count = '0;
    assign tr_drop = '0;
    assign tr_busy = 1'b0;
  end

  // ---- the slice (held in reset while RUN is 0) and the collective unit (one slice)
  logic core_rst;
  always_ff @(posedge clk) core_rst <= rst || !run;

  logic         coll_req, coll_ack, coll_gl;
  cmd_t         coll_cmd;
  logic [LANES-1:0]        coll_ren, coll_wen;
  logic [LANES-1:0][31:0]  coll_raddr, coll_rdata, coll_waddr, coll_wdata;
  cmd_t         coll_cmds [1];
  assign coll_cmds[0] = coll_cmd;

  otpu_slice #(.SID(0), .S(1), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS),
               .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
               .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
               .MXU_CL(MXU_CL), .VPU_CL(VPU_CL), .ULANES(ULANES), .PQ_WIN(PQ_WIN)) u_slice (
    .clk, .sys_rst(rst), .rst(core_rst), .ld_start, .ld_addr, .ld_n, .ld_busy,
    .a_rdy, .b_rdy, .sw_rdy, .wr_idle,
    .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
    .sw_req, .sw_addr, .sw_wdata, .sw_be,
    .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_rvalid, .b_rtag, .b_rdata,
    .coll_req, .coll_cmd, .coll_ack,
    .coll_ren, .coll_raddr, .coll_rdata,
    .coll_wen, .coll_waddr, .coll_wdata, .coll_gnt_local(coll_gl), .coll_gnt(coll_gl),
    .halted, .error, .icount, .pf, .dump(1'b0));

  otpu_coll #(.S(1), .LANES(LANES)) u_coll (
    .clk, .rst(core_rst), .req(coll_req), .cmds(coll_cmds), .gnt(coll_gl), .ack(coll_ack),
    .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
    .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata));

  // ---- memory
  logic [1:0][31:0]  awaddr, araddr;
  logic [1:0][511:0] wdata, rdata;
  logic [1:0][63:0]  wstrb;
  logic [1:0][1:0]   bresp, rresp;
  otpu_axi_dram #(.D(D), .BASE0(BASE0), .BASE1(BASE1)) u_mem (
    .clk, .rst,
    .a_rdy, .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
    .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
    .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_rvalid, .b_rtag, .b_rdata,
    .wr_idle,
    .m_awvalid(awvalid), .m_awready(awready), .m_awaddr(awaddr), .m_awid(awid),
    .m_wvalid(wvalid), .m_wready(wready), .m_wdata(wdata), .m_wstrb(wstrb),
    .m_bvalid(bvalid), .m_bready(bready), .m_bid(bid), .m_bresp(bresp),
    .m_arvalid(arvalid), .m_arready(arready), .m_araddr(araddr), .m_arid(arid),
    .m_rvalid(rvalid), .m_rready(rready), .m_rid(rid), .m_rdata(rdata), .m_rresp(rresp),
    .m_rlast(rlast), .err(axi_err));

  // single-beat, 64-byte, incrementing, normal non-cacheable bufferable
  assign {m0_axi_awlen, m1_axi_awlen, m0_axi_arlen, m1_axi_arlen} = '0;
  assign {m0_axi_awsize, m1_axi_awsize, m0_axi_arsize, m1_axi_arsize} = {4{3'd6}};
  assign {m0_axi_awburst, m1_axi_awburst, m0_axi_arburst, m1_axi_arburst} = {4{2'b01}};
  assign {m0_axi_awlock, m1_axi_awlock, m0_axi_arlock, m1_axi_arlock} = '0;
  assign {m0_axi_awcache, m1_axi_awcache, m0_axi_arcache, m1_axi_arcache} = {4{4'b0011}};
  assign {m0_axi_awprot, m1_axi_awprot, m0_axi_arprot, m1_axi_arprot} = '0;
  assign {m0_axi_awqos, m1_axi_awqos, m0_axi_arqos, m1_axi_arqos} = '0;
  assign m0_axi_wlast = 1'b1;
  assign m1_axi_wlast = 1'b1;

  assign m0_axi_awid = awid[0];     assign m1_axi_awid = awid[1];
  assign m0_axi_awaddr = awaddr[0]; assign m1_axi_awaddr = awaddr[1];
  assign m0_axi_awvalid = awvalid[0]; assign m1_axi_awvalid = awvalid[1];
  assign awready = {m1_axi_awready, m0_axi_awready};
  assign m0_axi_wdata = wdata[0];   assign m1_axi_wdata = wdata[1];
  assign m0_axi_wstrb = wstrb[0];   assign m1_axi_wstrb = wstrb[1];
  assign m0_axi_wvalid = wvalid[0]; assign m1_axi_wvalid = wvalid[1];
  assign wready = {m1_axi_wready, m0_axi_wready};
  assign bid = {m1_axi_bid, m0_axi_bid};
  assign bresp = {m1_axi_bresp, m0_axi_bresp};
  assign bvalid = {m1_axi_bvalid, m0_axi_bvalid};
  assign m0_axi_bready = bready[0]; assign m1_axi_bready = bready[1];
  assign m0_axi_arid = arid[0];     assign m1_axi_arid = arid[1];
  assign m0_axi_araddr = araddr[0]; assign m1_axi_araddr = araddr[1];
  assign m0_axi_arvalid = arvalid[0]; assign m1_axi_arvalid = arvalid[1];
  assign arready = {m1_axi_arready, m0_axi_arready};
  assign rid = {m1_axi_rid, m0_axi_rid};
  assign rdata = {m1_axi_rdata, m0_axi_rdata};
  assign rresp = {m1_axi_rresp, m0_axi_rresp};
  assign rlast = {m1_axi_rlast, m0_axi_rlast};
  assign rvalid = {m1_axi_rvalid, m0_axi_rvalid};
  assign m0_axi_rready = rready[0]; assign m1_axi_rready = rready[1];

  // ---- LEDs: heartbeat, running, halted/error
  logic [26:0] hb;
  always_ff @(posedge clk) hb <= hb + 1;
  assign led = {halted && !error, run && !halted, hb[26]};
endmodule
