// One openTPU slice: sequencer (dispatch window + scoreboard), DMA, MXU + ACT RAM, quantizer,
// VPU and TMEM. Units run concurrently. Shared resources are arbitrated every cycle:
//   TMEM   each bank serves 3 reads + 1 write per cycle; units are granted all-or-nothing in
//          the priority order DMA, COLL, VPU, MXU drain, QUANT (a unit that is not granted holds)
//   DRAM B DMA first, then the MXU stream (read responses are routed back by tag)
//   DRAM A the MXU scale stream (reads); QST writes have their own scalar write port (SW)
// The slice's DRAM sits outside (otpu_top) so that board wrappers can swap it. The DRAM may
// refuse requests (a_rdy/b_rdy, which must not depend on this cycle's requests), return reads
// after any latency (in order per port), and acknowledge writes late (wr_idle: none pending);
// rd_idle: no read of the slice's in flight (a run stopped by the host, rst, leaves its reads
// to come back after it).
//
// Loader: while the slice is held in reset (rst), ld_start copies ld_n instructions from DRAM
// byte address ld_addr (chunk aligned) into IMEM, one chunk (D/32 instructions) per cycle.
// It runs on sys_rst only, so the board can load a program and then release rst. Its reads
// start once the DRAM has no read in flight (rd_idle), so that no read of a stopped run, or of
// a load before, is taken for one of its rows; a LOAD restarts it. A HALT CHAIN reload that rst
// interrupts (the host stopped the run) stops.
//
// HALT CHAIN (docs/isa.md): when the sequencer halts with CHAIN and the DRAM has no write
// pending (wr_idle), the slice holds the sequencer and the units in reset (c_rst), loads the
// chained program with the loader and releases them: the run goes on with the run's arguments
// in R8..R15 and TMEM and DRAM as they were. `halted` stays low meanwhile, and icount counts
// every program of the run. RLD, a collective-unit instruction, is served here: one TMEM word
// through the collective's read port, converted, to the sequencer (never sent to otpu_coll).
module otpu_slice
  import otpu_pkg::*;
#(
  parameter int SID        = 0,
  parameter int S          = 1,
  parameter int D          = 32,
  parameter int MCOLS      = 8,
  parameter int ACT_BLOCKS = 64,
  parameter int ACT_ROWS   = MCOLS,   // ACT RAM rows (> MCOLS: the MXU replays chunks)
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 16,
  parameter int FIFO_DEPTH = 128,
  parameter int LANES      = 8,
  parameter int WIN        = 32,
  parameter int RPB        = 4,       // TMEM reads per bank per cycle
  parameter int WPB        = 2,       // TMEM writes per bank per cycle
  parameter int MXU_IMPL   = 0,       // MXU dot product: 0 adder tree, 1 DSP cascade chains, 2 systolic
  parameter int MXU_CL     = 16,      // cascade chain length
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,  // VPU lanes with the composite functions
  parameter int ULANES     = LANES,   // TMEM lanes of the MXU and the quantizer (<= LANES)
  parameter int PQ_WIN     = 64,      // cycles per P/Q counter window (+bucket= in simulation)
  parameter bit HAS_DSTEP  = 1'b1     // streams (DSTEP, STREAM): SE's tail in the VPU
) (
  input  logic          clk,
  input  logic          sys_rst,
  input  logic          rst,
  input  logic [31:0]   rinit [8],  // the run's arguments: R8..R15 at the start (otpu_seq)
  // program loader
  input  logic          ld_start,
  input  logic [31:0]   ld_addr,
  input  logic [31:0]   ld_n,
  output logic          ld_busy,
  // DRAM
  input  logic          a_rdy,
  input  logic          b_rdy,
  input  logic          sw_rdy,
  input  logic          wr_idle,
  input  logic          rd_idle,    // no read in flight (the loader waits for it)
  output logic          a_req,
  output logic          a_we,
  output logic [31:0]   a_addr,
  output logic [31:0]   a_wdata,
  output logic [3:0]    a_be,
  output logic          sw_req,     // scalar (QST) writes: one byte-enabled word per cycle
  output logic [31:0]   sw_addr,
  output logic [31:0]   sw_wdata,
  output logic [3:0]    sw_be,
  input  logic          a_rvalid,
  input  logic [31:0]   a_rdata,
  input  logic [31:0]   a_rdata2,   // the other word of a_rdata's 8-byte pair (MM PAIR scales)
  output logic          b_req,
  output logic          b_tag,
  output logic          b_we,
  output logic [D/4-1:0] b_wmask,
  output logic [D*8-1:0] b_wdata,
  output logic [31:0]   b_addr,
  output logic          b_par,      // ^b_addr[31:5] (the memory's channel hash; see the port mux)
  input  logic          b_rvalid,
  input  logic          b_rtag,
  input  logic [D*8-1:0] b_rdata,
  // collective
  output logic                    coll_req,
  output cmd_t                    coll_cmd,
  input  logic                    coll_ack,
  input  logic [LANES-1:0]        coll_ren,
  input  logic [LANES-1:0][31:0]  coll_raddr,
  output logic [LANES-1:0][31:0]  coll_rdata,
  input  logic [LANES-1:0]        coll_wen,
  input  logic [LANES-1:0][31:0]  coll_waddr,
  input  logic [LANES-1:0][31:0]  coll_wdata,
  output logic                    coll_gnt_local,
  input  logic                    coll_gnt,
  // status
  output logic          halted,
  output logic          error,
  output logic          wait_to,    // the error is a WAITW timeout (STATUS WAIT_TO)
  output logic          a_inval,    // a WAITW held: port A's held beats may be stale (otpu_dma ww_held)
  output logic [31:0]   icount,
  output perf_t         pf,         // activity and trace events, a cycle late (otpu_pkg)
  input  logic          dump
);
  localparam int BW = $clog2(LANES);
  // the stream engine (docs/stream.md): the DMA moves a stream (DSTEP, STREAM) through SE in
  // the VPU; streams need 8 lanes on both sides (otpu_dma W = 8: D >= 32). Without it DSTEP and
  // STREAM are illegal instructions (otpu_seq STREAMS: the slice stops with ERROR)
  localparam bit HAS_SS = HAS_DSTEP && LANES == 8 && D >= 32;
  localparam int P_MXU = 0, P_DMA = 1, P_Q = 2, P_VA = 3, P_VB = 4, P_Q3 = 5, P_Q2 = 6, P_COLL = 7, NRP = 8;
  // the board build (TMEM replicated per read port, RPB >= NRP * LANES: reads never conflict;
  // WPB = 1): the arbiter only masks write banks (see the arbiter), and the VPU's grant only
  // takes its buffered writes (otpu_vpu WBUF)
  localparam bit ARB_MASK = (RPB >= NRP * LANES) && (WPB == 1);
  localparam int W_DMA = 0, W_MXU = 1, W_COLL = 2, W_VPU = 3, NWP = 4;
  // The VPU ahead of the MXU drain: the VPU writes all its lanes' banks at once, so an MXU drain
  // write in any one bank cost it the whole cycle, and decode's softmax (EXP2SUB, which the next
  // PV MM waits for) ran at about half its rate; the drain holds a cycle instead
  // (docs/litedram.md section 11, "The core's own gaps": Qwen3 4-bit -3.0% cycles per token).
  localparam int G_DMA = 0, G_COLL = 1, G_VPU = 2, G_MXU = 3, G_Q = 4, NG = 5;

  // ---- sequencer
  cmd_t ucmd [NUNITS];
  logic [NUNITS-1:0] ustart, urdy, udone;
  logic urel;
  logic im_we;
  logic [31:0] im_row;
  seq_ev_t sq_ev;
  logic        srst;              // the sequencer's and the units' reset: rst, a CHAIN reload, or
                                  //   a WAITW timeout (to_err)
  logic        sq_halted, sq_ch, sq_err, rl_v;
  logic        dma_err;           // a WAITW timed out: the slice stops with an error (to_err)
  logic [31:0] sq_icount, sq_cha, sq_chn, rl_val;
  otpu_seq #(.IMEM_WORDS(IMEM_WORDS), .SID(SID), .S(S), .D(D), .WIN(WIN), .STREAMS(HAS_SS)) u_seq (
    .clk, .rst(srst), .ucmd, .ustart, .urel, .urdy, .udone, .halted(sq_halted), .error(sq_err),
    .icount(sq_icount), .ev(sq_ev), .rinit, .rl_v, .rl_val, .ch_req(sq_ch), .ch_addr(sq_cha),
    .ch_n(sq_chn), .im_we, .im_row, .im_data(b_rdata));

  // ---- HALT CHAIN: reload IMEM from the chained program's address, the units held in reset.
  // A WAITW timeout (dma_err) stops the slice until rst: to_err holds the sequencer and the
  // units in reset, so no instruction starts or ends after it (TMEM, DRAM and ICOUNT stay as
  // they were), and HALTED rises once they are in reset (to_q: the units' resets are up to two
  // registers behind srst) and the DRAM has taken every write they gave it (wr_idle). ICOUNT
  // keeps the sequencer's count as it stood (ic_base, as at a CHAIN reload)
  logic        c_rst, c_ld, to_err;
  logic [2:0]  to_q;
  logic [31:0] c_addr, c_n, ic_base;
  assign srst = rst || c_rst || to_err;
  assign halted = (sq_halted && !sq_ch) || (to_q[2] && wr_idle);
  assign error = sq_err || to_err;
  assign wait_to = to_err;
  assign icount = ic_base + sq_icount;
  always_ff @(posedge clk) begin
    c_ld <= 1'b0;
    to_q <= rst ? 3'b000 : {to_q[1:0], to_err};
    if (rst) begin
      c_rst <= 1'b0; ic_base <= '0; to_err <= 1'b0;
    end else if (dma_err || to_err) begin
      to_err <= 1'b1;
      if (to_err && !to_q[0]) ic_base <= ic_base + sq_icount;   // its last cycle out of reset
    end else if (!c_rst) begin
      if (sq_halted && sq_ch && wr_idle) begin
        c_rst <= 1'b1; c_ld <= 1'b1;
        c_addr <= sq_cha; c_n <= sq_chn;
        ic_base <= ic_base + sq_icount;
      end
    end else if (!c_ld && !ld_busy) c_rst <= 1'b0;      // loaded (the loader starts after c_ld)
  end

  // ---- program loader (see the top): ld_go once the DRAM had no read in flight; ld_ch: a HALT
  // CHAIN reload (rst stops it). While busy the loader has port B to itself
  localparam int IPR = D / 32;
  logic [31:0] ld_rows, ld_iss, ld_cmp, ld_a;
  logic        ld_go, ld_ch;
  wire ld_req = ld_busy && ld_go && ld_iss < ld_rows;
  wire [31:0] ld_n_s = ld_start ? ld_n : c_n;
  assign im_we  = ld_busy && ld_go && b_rvalid;
  assign im_row = ld_cmp;
  always_ff @(posedge clk) begin
    if (sys_rst) begin
      ld_busy <= 1'b0;
    end else if (ld_start || (c_ld && !ld_busy)) begin
      ld_rows <= (ld_n_s + IPR - 1) / IPR;
      ld_iss <= '0; ld_cmp <= '0;
      ld_a <= (ld_start ? ld_addr : c_addr) >> 2;
      ld_busy <= (ld_n_s != 0);
      ld_go <= 1'b0;
      ld_ch <= !ld_start;
    end else if (ld_busy) begin
      if (rst && ld_ch) ld_busy <= 1'b0;
      else if (!ld_go) ld_go <= rd_idle;
      else begin
        if (ld_req && b_rdy) begin
          ld_iss <= ld_iss + 1;
          ld_a <= ld_a + D / 4;
        end
        if (b_rvalid) begin
          ld_cmp <= ld_cmp + 1;
          if (ld_cmp + 1 == ld_rows) ld_busy <= 1'b0;
        end
      end
    end
  end

  // ---- TMEM
  logic [NRP-1:0][LANES-1:0]       rq_en, r_en;
  logic [NRP-1:0][LANES-1:0][31:0] r_addr, r_data;
  logic [NWP-1:0][LANES-1:0]       wq_en, w_en;
  logic [NWP-1:0][LANES-1:0][31:0] w_addr, w_data;
  logic [NWP-1:0]                  w_gnt;   // each write port's grant (the DMA always writes)
  // the MXU's ports keep the crossbar (its drain writes scattered rows); the others are rotators.
  // The quantizer's row-factor port (one word) and the collective's read port, rarely busy,
  // share the DMA's copy (ST reads): 6 copies of the memory instead of 8.
  otpu_tmem #(.WORDS(TMEM_WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(WPB),
              .GEN_R(NRP'(1) << P_MXU), .GEN_W(NWP'(1) << W_MXU),
              .SH_HOST(P_DMA), .SH_MASK((NRP'(1) << P_Q3) | (NRP'(1) << P_COLL)),
              .SID(SID)) u_tmem (
    .clk, .r_en, .r_req(rq_en), .r_addr, .r_data, .w_en, .w_req(wq_en), .w_gnt, .w_addr,
    .w_data, .dump);
  assign coll_rdata = r_data[P_COLL];
  // the units' TMEM requests
  logic [LANES-1:0]        dma_ren, dma_wen, va_ren, vb_ren, v_wen;
  logic                    v_ren;   // the VPU's lanes take their read data this cycle
  logic [LANES-1:0][31:0]  dma_raddr, dma_waddr, dma_wdata, va_raddr, vb_raddr, v_waddr, v_wdata;
  logic [ULANES-1:0]       mxu_ren, mxu_wen, q_ren, q_ren2;
  logic [ULANES-1:0][31:0] mxu_raddr, mxu_waddr, mxu_wdata, q_raddr, q_raddr2;

  // ---- ACT RAM
  logic [ULANES-1:0]      act_we;
  logic [ULANES-1:0][7:0] act_data;
  logic                   asc_we;
  logic [7:0]             act_row, asc_row, act_off;
  logic                   act_dup;
  logic [31:0]            act_idx, asc_data;
  logic [15:0]            asc_blk, act_rblk, act_rblk2;
  logic [MCOLS-1:0]       act_rhi;
  logic [7:0]             act_rgrp;
  logic                   act_ren;
  logic                   mxu_pop;       // the MXU consumes a block this cycle (its ACT read)
  logic [MCOLS*D*8-1:0]   act_rdata;
  logic [MCOLS*32-1:0]    act_rscale;
  otpu_actram #(.D(D), .MCOLS(MCOLS), .ROWS(ACT_ROWS), .BLOCKS(ACT_BLOCKS), .LANES(ULANES)) u_act (
    .clk, .we(act_we), .w_row(act_row), .w_idx(act_idx), .w_data(act_data), .w_dup(act_dup),
    .w_off(act_off), .swe(asc_we),
    .s_row(asc_row), .s_blk(asc_blk), .s_data(asc_data), .ren(act_ren), .r_use(mxu_pop), .r_blk(act_rblk),
    .r_blk2(act_rblk2), .r_hi(act_rhi), .r_grp(act_rgrp),
    .r_data(act_rdata), .r_scale(act_rscale));

  // ---- units
  logic [NG-1:0] gnt;
  logic        q3_en;
  logic [31:0] q3_addr;
  logic d_dma, d_mxu, d_q, d_vpu, r_dma, r_mxu, r_q, r_vpu;
  logic mxu_areq, q_areq, q_awant, q_awe, mxu_agnt, mxu_bgnt;
  logic [31:0] mxu_aaddr, q_aaddr, q_awdata;
  logic [3:0] q_abe;
  logic dma_breq, dma_bwe, mxu_breq;
  logic [D/4-1:0] dma_bwmask;
  logic [D*8-1:0] dma_bwdata;
  logic [31:0] dma_baddr, mxu_baddr;
  localparam int FW = $clog2(FIFO_DEPTH) + 1;
  logic [FW-1:0] mxu_level;
  logic mxu_starve, mxu_block, mxu_u, q_u, v_u;
  logic [3:0][31:0] mxu_uv;
  logic [31:0] q_frz, v_frz;

  // Each unit's reset is its own register (a reset tree): the slice's reset reaches ~15k flip-
  // flops across the die, and one replicated net from the board ran 8.4 ns routes into the
  // units (the worst core_clk paths at 114 MHz). The units leave reset a cycle after the
  // sequencer, which starts nothing that early. max_fanout 64: at 125.49 MHz rst_vpu (256 per
  // copy) into the VPU lanes' c registers had 0.25 ns slack. Q, the VPU and the MXU take a
  // second stage (rst_q0, rst_v0, rst_m0), so their replicas are loaded next to them: in the SE
  // v2 build the board's core_rst copies into rst_vpu / rst_q had -0.107 ns with no logic, on
  // 812bb01 rst_mxu into the MXU's weight FIFO pointers +0.123 ns with none. They leave reset
  // two cycles after the sequencer (checked below).
  (* max_fanout = 64 *) logic rst_dma, rst_mxu, rst_q, rst_vpu;
  logic rst_q0, rst_v0, rst_m0;
  always_ff @(posedge clk) begin
    rst_dma <= srst; rst_m0 <= srst; rst_q0 <= srst; rst_v0 <= srst;
    rst_mxu <= rst_m0; rst_q <= rst_q0; rst_vpu <= rst_v0;
  end
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if ((ustart[U_DMA] && rst_dma) || (ustart[U_MXU] && rst_mxu) || (ustart[U_Q] && rst_q) ||
        (ustart[U_VPU] && rst_vpu))
      $fatal(1, "otpu_slice: a unit started while in reset (ustart %b)", ustart);
`endif

  // the stream engine (HAS_SS, above)
  logic        ss_req, ss_gnt, ss_pe, ss_in_v, ss_y_v, ss_o_v;
  ss_cfg_t     ss_cfg;
  logic [2:0]  ss_fk;
  logic [4:0]  ss_fi;
  logic [31:0] ss_in_d [LANES], ss_fd [LANES], ss_y_d [LANES];
  logic [31:0] ss_o_d;

  otpu_dma #(.D(D), .LANES(LANES), .HAS_DSTEP(HAS_SS)) u_dma (
    .clk, .rst(rst_dma), .start(ustart[U_DMA]), .cmd(ucmd[U_DMA]), .rdy(r_dma), .done(d_dma),
    .err(dma_err), .ww_held(a_inval),
    .b_req(dma_breq), .b_gnt(b_rdy), .b_we(dma_bwe), .b_wmask(dma_bwmask), .b_wdata(dma_bwdata),
    .b_addr(dma_baddr), .b_rvalid(b_rvalid && b_rtag), .b_rdata, .wr_idle,
    .t_ren(dma_ren), .t_raddr(dma_raddr), .t_rdata(r_data[P_DMA]),
    .t_wen(dma_wen), .t_waddr(dma_waddr), .t_wdata(dma_wdata),
    .ss_req, .ss_gnt, .ss_cfg, .ss_pe, .ss_in_v, .ss_in_d, .ss_fk, .ss_fi, .ss_fd,
    .ss_y_v, .ss_y_d, .ss_o_v, .ss_o_d);

  otpu_mxu #(.D(D), .MCOLS(MCOLS), .ROWS(ACT_ROWS), .DEPTH(FIFO_DEPTH), .LANES(ULANES), .IMPL(MXU_IMPL),
             .CL(MXU_CL), .SID(SID)) u_mxu (
    .clk, .rst(rst_mxu), .start(ustart[U_MXU]), .go(urel), .cmd(ucmd[U_MXU]), .rdy(r_mxu), .done(d_mxu),
    .computing(mxu_pop), .pf_level(mxu_level), .pf_starve(mxu_starve), .pf_block(mxu_block),
    .pf_u(mxu_u), .pf_uv(mxu_uv),
    .act_blk(act_rblk), .act_blk2(act_rblk2), .act_hi(act_rhi), .act_grp(act_rgrp), .act_ren, .act_data(act_rdata), .act_scale(act_rscale),
    .a_req(mxu_areq), .a_addr(mxu_aaddr), .a_gnt(mxu_agnt), .a_rvalid, .a_rdata, .a_rdata2,
    .b_req(mxu_breq), .b_addr(mxu_baddr), .b_gnt(mxu_bgnt), .b_rvalid(b_rvalid && !b_rtag),
    .b_rdata,
    .t_ren(mxu_ren), .t_raddr(mxu_raddr), .t_rdata(r_data[P_MXU][ULANES-1:0]),
    .t_wen(mxu_wen), .t_waddr(mxu_waddr), .t_wdata(mxu_wdata), .t_gnt(gnt[G_MXU]));

  otpu_quant #(.D(D), .LANES(ULANES), .SID(SID)) u_quant (
    .clk, .rst(rst_q), .start(ustart[U_Q]), .cmd(ucmd[U_Q]), .rdy(r_q), .done(d_q), .gnt(gnt[G_Q]),
    .t_ren(q_ren), .t_raddr(q_raddr), .t_rdata(r_data[P_Q][ULANES-1:0]),
    .t_ren2(q_ren2), .t_raddr2(q_raddr2), .t_rdata2(r_data[P_Q2][ULANES-1:0]),
    .t_ren3(q3_en), .t_raddr3(q3_addr), .t_rdata3(r_data[P_Q3][0]),
    .act_we, .act_row, .act_idx, .act_data, .act_dup, .act_off, .asc_we, .asc_row, .asc_blk, .asc_data,
    .a_want(q_awant), .wr_idle,
    .a_req(q_areq), .a_we(q_awe), .a_addr(q_aaddr), .a_wdata(q_awdata), .a_be(q_abe),
    .pf_u(q_u), .pf_frz(q_frz));

  otpu_vpu #(.LANES(LANES), .CL(VPU_CL), .SID(SID), .WBUF(ARB_MASK), .HAS_SE(HAS_SS)) u_vpu (
    .clk, .rst(rst_vpu), .start(ustart[U_VPU]), .cmd(ucmd[U_VPU]), .rdy(r_vpu), .done(d_vpu),
    .gnt(gnt[G_VPU]), .ren(v_ren),
    .ta_en(va_ren), .ta_addr(va_raddr), .ta_data(r_data[P_VA]),
    .tb_en(vb_ren), .tb_addr(vb_raddr), .tb_data(r_data[P_VB]),
    .tw_en(v_wen), .tw_addr(v_waddr), .tw_data(v_wdata), .pf_u(v_u), .pf_frz(v_frz),
    .ss_req, .ss_gnt, .ss_cfg, .ss_pe, .ss_in_v, .ss_in_d, .ss_fk, .ss_fi, .ss_fd,
    .ss_y_v, .ss_y_d, .ss_o_v, .ss_o_d);

  // collective: request from start until acknowledged (an RLD is not a collective)
  wire rl_start = ustart[U_COLL] && ucmd[U_COLL].op == OP_RLD;
  always_ff @(posedge clk) begin
    if (srst) coll_req <= 1'b0;
    else if (ustart[U_COLL] && !rl_start) coll_req <= 1'b1;
    else if (coll_ack) coll_req <= 1'b0;
  end
  assign coll_cmd = ucmd[U_COLL];
  // RLD: the word through the collective's read port (lane 0) once granted; the lane has it
  // the next cycle (rl_got), registered (rl_cv), converted (rl_c1), times the multiplier
  // (1 without MUL) as two partial products (rl_c2: the low 32 bits of x * m[15:0] and the
  // low 16 of x[15:0] * m[31:16]), summed and handed to the sequencer (rl_v)
  logic        rl_pend, rl_got, rl_cv, rl_c1, rl_c2, rl_raw;
  logic [31:0] rl_a, rl_w, rl_m, rl_x, rl_pl;
  logic [15:0] rl_ph;
  wire         rl_busy = rl_pend || rl_got || rl_cv || rl_c1 || rl_c2 || rl_v;
  always_ff @(posedge clk) begin
    if (srst) begin
      rl_pend <= 1'b0; rl_got <= 1'b0; rl_cv <= 1'b0; rl_c1 <= 1'b0; rl_c2 <= 1'b0;
      rl_v <= 1'b0;
    end else begin
      rl_got <= rl_pend && r_en[P_COLL][0];
      if (rl_start) begin
        rl_pend <= 1'b1;
        rl_a <= ucmd[U_COLL].w1;
        rl_raw <= ucmd[U_COLL].flags[RF_RAW];
        rl_m <= ucmd[U_COLL].flags[RF_MUL] ? ucmd[U_COLL].w2 : 32'd1;
      end else if (rl_pend && r_en[P_COLL][0]) rl_pend <= 1'b0;
      rl_cv <= rl_got;
      if (rl_got) rl_w <= r_data[P_COLL][0];
      rl_c1 <= rl_cv;
      if (rl_cv) rl_x <= rl_raw ? rl_w : rld_f2i(rl_w);
      rl_c2 <= rl_c1;
      if (rl_c1) begin
        rl_pl <= rl_x * rl_m[15:0];
        rl_ph <= rl_x[15:0] * rl_m[31:16];
      end
      rl_v <= rl_c2;
      if (rl_c2) rl_val <= rl_pl + {rl_ph, 16'd0};
    end
  end
  // the TMEM request arrays, each assembled in one process from the units' ports
  always_comb begin
    rq_en = '0; r_addr = '0; wq_en = '0; w_addr = '0; w_data = '0;
    rq_en[P_DMA] = dma_ren;   r_addr[P_DMA] = dma_raddr;
    rq_en[P_MXU][ULANES-1:0] = mxu_ren;   r_addr[P_MXU][ULANES-1:0] = mxu_raddr;
    rq_en[P_Q][ULANES-1:0] = q_ren;       r_addr[P_Q][ULANES-1:0] = q_raddr;
    rq_en[P_Q2][ULANES-1:0] = q_ren2;     r_addr[P_Q2][ULANES-1:0] = q_raddr2;
    rq_en[P_Q3][0] = q3_en;   r_addr[P_Q3][0] = q3_addr;
    rq_en[P_VA] = va_ren;     r_addr[P_VA] = va_raddr;
    rq_en[P_VB] = vb_ren;     r_addr[P_VB] = vb_raddr;
    rq_en[P_COLL] = coll_ren; r_addr[P_COLL] = coll_raddr;
    if (rl_pend) begin rq_en[P_COLL][0] = 1'b1; r_addr[P_COLL][0] = rl_a; end
    wq_en[W_DMA] = dma_wen;   w_addr[W_DMA] = dma_waddr;   w_data[W_DMA] = dma_wdata;
    wq_en[W_MXU][ULANES-1:0] = mxu_wen;
    w_addr[W_MXU][ULANES-1:0] = mxu_waddr;
    w_data[W_MXU][ULANES-1:0] = mxu_wdata;
    wq_en[W_VPU] = v_wen;     w_addr[W_VPU] = v_waddr;     w_data[W_VPU] = v_wdata;
    wq_en[W_COLL] = coll_wen; w_addr[W_COLL] = coll_waddr; w_data[W_COLL] = coll_wdata;
  end

  assign urdy  = {!coll_req && !rl_busy, r_vpu, r_q, r_mxu, r_dma};
  assign udone = {coll_ack || rl_v, d_vpu, d_q, d_mxu, d_dma};

  // ---- TMEM bank arbiter: all-or-nothing grants in priority order
  function automatic logic [NRP-1:0] grp_rports(input int g);
    case (g)
      G_DMA:  return NRP'(1) << P_DMA;
      G_COLL: return NRP'(1) << P_COLL;
      G_MXU:  return NRP'(1) << P_MXU;
      G_Q:    return (NRP'(1) << P_Q) | (NRP'(1) << P_Q2) | (NRP'(1) << P_Q3);
      default: return (NRP'(1) << P_VA) | (NRP'(1) << P_VB);
    endcase
  endfunction
  function automatic logic [NWP-1:0] grp_wports(input int g);
    case (g)
      G_DMA:  return NWP'(1) << W_DMA;
      G_COLL: return NWP'(1) << W_COLL;
      G_MXU:  return NWP'(1) << W_MXU;
      G_Q:    return '0;
      default: return NWP'(1) << W_VPU;
    endcase
  endfunction

  // The board build (TMEM replicated per read port, RPB >= NRP * LANES: reads never conflict;
  // WPB = 1; ARB_MASK) only needs write-bank masks: a unit is granted when no bank it writes was taken by
  // a higher-priority unit this cycle (a unit's own lanes write distinct banks). Shallow logic,
  // no counters. Other configurations count reads and writes per bank.
  logic [NG-1:0] gnt_m, gnt_c;
  logic          cgl_m, cgl_c;
  // Each group's write-bank mask, the pairwise conflicts between groups (in parallel), then
  // priority on single bits: a group is admitted unless it conflicts with an earlier admitted
  // group (the same result as accumulating the admitted groups' banks in priority order, with
  // the per-bank reductions out of the priority chain)
  logic [NG-1:0][LANES-1:0] amk;
  logic [NG-1:0][NG-1:0]    acf;
  logic [NG-1:0]            aok;
  always_comb begin
    logic [NWP-1:0] wp;
    for (int g = 0; g < NG; g++) begin
      wp = grp_wports(g);
      amk[g] = '0;
      for (int p = 0; p < NWP; p++)
        if (wp[p])
          for (int l = 0; l < LANES; l++)
            if (wq_en[p][l]) amk[g][w_addr[p][l][BW-1:0]] = 1'b1;
    end
    acf = '0;
    for (int g = 0; g < NG; g++)
      for (int h = 0; h < g; h++) acf[g][h] = (amk[g] & amk[h]) != '0;
  end
  // the guests on the DMA's TMEM copy read only when no port before them there asks (from the
  // requests: the copy's block RAM address does not wait for the grants)
  wire q3_blk = (|rq_en[P_Q3]) && (|rq_en[P_DMA]);
  wire coll_blk = (|rq_en[P_COLL]) && ((|rq_en[P_DMA]) || (|rq_en[P_Q3]));
  always_comb begin
    for (int g = 0; g < NG; g++) begin
      aok[g] = 1'b1;
      for (int h = 0; h < g; h++) if (aok[h] && acf[g][h]) aok[g] = 1'b0;
      if (g == G_Q && q_awant && !sw_rdy) aok[g] = 1'b0;   // QST write the DRAM cannot take
      if ((g == G_Q && q3_blk) || (g == G_COLL && coll_blk)) aok[g] = 1'b0;
    end
    gnt_m = aok;
    gnt_m[G_COLL] = coll_gnt;
    cgl_m = aok[G_COLL];
  end

  always_comb begin
    gnt = ARB_MASK ? gnt_m : gnt_c;
    coll_gnt_local = ARB_MASK ? cgl_m : cgl_c;
    for (int p = 0; p < NRP; p++) r_en[p] = rq_en[p];
    for (int p = 0; p < NWP; p++) w_en[p] = wq_en[p];
    w_gnt = '1;
    w_gnt[W_MXU] = gnt[G_MXU];
    w_gnt[W_VPU] = gnt[G_VPU];
    w_gnt[W_COLL] = gnt[G_COLL];
    if (!gnt[G_MXU])  begin r_en[P_MXU] = '0; w_en[W_MXU] = '0; end
    if (!gnt[G_Q])    begin r_en[P_Q] = '0; r_en[P_Q2] = '0; r_en[P_Q3] = '0; end
    if (!v_ren)       begin r_en[P_VA] = '0; r_en[P_VB] = '0; end
    if (!gnt[G_VPU])  w_en[W_VPU] = '0;
    if (!gnt[G_COLL]) begin r_en[P_COLL] = '0; w_en[W_COLL] = '0; end
  end

  always_comb begin
    int rc [LANES];
    int wc [LANES];
    int ur [LANES];
    int uw [LANES];
    logic ok;
    logic [NRP-1:0] rp;
    logic [NWP-1:0] wp;
    for (int b = 0; b < LANES; b++) begin rc[b] = 0; wc[b] = 0; end
    cgl_c = 1'b1;
    for (int g = 0; g < NG; g++) begin
      rp = grp_rports(g);
      wp = grp_wports(g);
      for (int b = 0; b < LANES; b++) begin ur[b] = 0; uw[b] = 0; end
      for (int p = 0; p < NRP; p++)
        if (rp[p])
          for (int l = 0; l < LANES; l++)
            if (rq_en[p][l]) ur[r_addr[p][l][BW-1:0]] += 1;
      for (int p = 0; p < NWP; p++)
        if (wp[p])
          for (int l = 0; l < LANES; l++)
            if (wq_en[p][l]) uw[w_addr[p][l][BW-1:0]] += 1;
      ok = 1'b1;
      for (int b = 0; b < LANES; b++)
        if (rc[b] + ur[b] > RPB || wc[b] + uw[b] > WPB) ok = 1'b0;
      if (g == G_Q && q_awant && !sw_rdy) ok = 1'b0;    // QST write the DRAM cannot take
      if ((g == G_Q && q3_blk) || (g == G_COLL && coll_blk)) ok = 1'b0;
      if (g == G_COLL) begin
        cgl_c = ok;
        gnt_c[g] = coll_gnt;      // every slice must grant the collective
      end else begin
        gnt_c[g] = ok;
      end
      if (ok)
        for (int b = 0; b < LANES; b++) begin rc[b] += ur[b]; wc[b] += uw[b]; end
    end
  end

  // ---- DRAM ports
  assign mxu_bgnt = !dma_breq && b_rdy;
  assign mxu_agnt = a_rdy;
  always_comb begin
    a_req    = mxu_areq;
    a_we     = 1'b0;
    a_addr   = mxu_aaddr;
    a_wdata  = '0;
    a_be     = '0;
    sw_req   = q_areq && q_awe;
    sw_addr  = q_aaddr;
    sw_wdata = q_awdata;
    sw_be    = q_abe;
    b_req   = dma_breq | mxu_breq;
    b_tag   = dma_breq;
    b_we    = dma_bwe;
    b_wmask = dma_bwmask;
    b_wdata = dma_bwdata;
    b_addr  = dma_breq ? dma_baddr : mxu_baddr;
    // the address's parity from each source's own address (their registers), muxed with it:
    // the parity of the muxed address ran on from the DMA's request and the MXU's grant into
    // every write of otpu_native_dram's queues (DMA occ -> qbm, 11 levels, the 959b425 build's
    // worst DMA -> memory paths, -0.390 ns at 133.33 MHz)
    b_par   = dma_breq ? ^dma_baddr[31:5] : ^mxu_baddr[31:5];
    if (ld_busy) begin                // the units are held in reset
      b_req = ld_req; b_tag = 1'b1; b_we = 1'b0; b_addr = ld_a; b_par = ^ld_a[31:5];
    end
  end

  // ---- activity and trace events (pf; docs/observability.md). The sequencer's events come
  // registered (seq_ev_t), the units report their counters the cycle after an instruction ends,
  // and the port and stall counters of the H, P and Q trace lines live here:
  //   P  per window of `win` cycles: DRAM port B requests (MXU bm, DMA bd), port A requests
  //      (MXU am, QST aq), MXU compute (mx), and cycles the MXU drain, QUANT, VPU and collective
  //      lost to TMEM bank arbitration (fm fq fv fc)
  //   Q  the same window: a port B / A request waited for the memory (bs, as), the MXU had work
  //      but no chunk (ms) or chunks but did not consume (mb), the summed MXU chunk-FIFO level
  //      (ff; average = ff / n), the program loader used port B (ld)
  //   H  at the halt: the port requests since reset (bmxu bdma amxu aq)
  // Every cycle's activity is registered first (a*_r) and the sums run a cycle behind, so no
  // counter adds to an arbitration or DRAM-ready path: a window's sums are s + a_r the cycle
  // after its last cycle (pf.w), the halt totals the cycle after the halt (pf.h). Counting
  // stops the cycle after the halt. In simulation, +trace prints the trace lines from pf.
  // a unit "loses" a cycle when it requested TMEM ports and was not granted
  wire lose_mxu = !gnt[G_MXU] && ((|rq_en[P_MXU]) || (|wq_en[W_MXU]));
  wire lose_q   = !gnt[G_Q] && ((|rq_en[P_Q]) || (|rq_en[P_Q2]) || q3_en);
  wire lose_vpu = !gnt[G_VPU] && ((|rq_en[P_VA]) || (|rq_en[P_VB]) || (|wq_en[W_VPU]));
  wire lose_col = !gnt[G_COLL] && ((|rq_en[P_COLL]) || (|wq_en[W_COLL]));

  logic [31:0] win;                  // cycles per window
`ifdef SYNTHESIS
  assign win = 32'(PQ_WIN);
`else
  int bucket;
  initial begin
    if (!$value$plusargs("bucket=%d", bucket)) bucket = PQ_WIN;
    if (bucket < 1 || bucket >= (1 << 24)) $fatal(1, "otpu_slice: bucket must be 1 .. 2^24 - 1");
  end
  assign win = 32'(bucket);
`endif
  initial if (PQ_WIN < 1 || PQ_WIN >= (1 << 24)) $fatal(1, "otpu_slice: PQ_WIN out of range");

  // this cycle's activity: the P fields, the Q fields but ff (bs as ms mb ld), the FIFO level
  logic [NP-1:0] ap, ap_r;
  logic [4:0]    aq, aq_r;
  logic [FW-1:0] af_r;
  assign ap = {lose_col, lose_vpu, lose_q, lose_mxu, mxu_pop, q_areq, mxu_areq, dma_breq,
               mxu_breq};
  assign aq = {ld_busy && ld_req, mxu_block, mxu_starve, (a_req || q_awant) && !a_rdy,
               b_req && !b_rdy};
  logic [31:0] w_n, n_r;             // cycles of the current window before this one; last n
  logic        h_d, we_r, h_ev;      // halted a cycle ago; a window ended / the halt, last cycle
  logic [NP-1:0][31:0] sp;           // window sums, a cycle behind
  logic [NQ-1:0][31:0] sq;
  logic [NH-1:0][31:0] sh;           // totals since reset, a cycle behind
  wire w_end = !h_d && (w_n + 1 == win || halted);
  always_ff @(posedge clk) begin
    if (rst) begin
      h_d <= 1'b0; we_r <= 1'b0; h_ev <= 1'b0; w_n <= '0;
      ap_r <= '0; aq_r <= '0; af_r <= '0;
      sp <= '0; sq <= '0; sh <= '0;
    end else begin
      h_d <= halted;
      h_ev <= halted && !h_d;
      we_r <= w_end;
      n_r <= w_n + 1;
      if (!h_d) w_n <= w_end ? '0 : w_n + 1;
      ap_r <= h_d ? '0 : ap;
      aq_r <= h_d ? '0 : aq;
      af_r <= h_d ? '0 : mxu_level;
      for (int k = 0; k < NP; k++) sp[k] <= we_r ? '0 : sp[k] + 32'(ap_r[k]);
      for (int k = 0; k < NQ; k++)
        sq[k] <= we_r ? '0 : sq[k] + ((k == 4) ? 32'(af_r) : 32'(aq_r[k < 4 ? k : 4]));
      for (int k = 0; k < NH; k++) sh[k] <= sh[k] + 32'(ap_r[k]);
    end
  end

  always_comb begin
    pf.sq = sq_ev;
    pf.mac = ap_r[4];
    pf.starve = aq_r[2];
    pf.deny = |ap_r[8:5];
    pf.u_mxu = mxu_u;
    pf.u_mxu_v = mxu_uv;
    pf.u_q = q_u;
    pf.u_q_frz = q_frz;
    pf.u_vpu = v_u;
    pf.u_vpu_frz = v_frz;
    pf.w = we_r;
    pf.w_n = n_r;
    for (int k = 0; k < NP; k++) pf.w_p[k] = sp[k] + 32'(ap_r[k]);
    for (int k = 0; k < NQ; k++)
      pf.w_q[k] = sq[k] + ((k == 4) ? 32'(af_r) : 32'(aq_r[k < 4 ? k : 4]));
    pf.h = h_ev;
    pf.h_v = sh;
  end

`ifndef SYNTHESIS
  // ---- +trace: the trace lines (opentpu/profile.py), from the events above -- exactly what the
  // board's trace buffer records (otpu_trace.sv, opentpu/hwtrace.py)
  bit trace;
  initial trace = $test$plusargs("trace");
  always_ff @(posedge clk)
    if (trace) begin
      if (pf.sq.g) $display("T%0d G c=%0d s=%0d", SID, pf.sq.cyc, pf.sq.g_slot);
      for (int i = 0; i < 32; i++)
        if (pf.sq.e[i]) $display("T%0d E c=%0d s=%0d", SID, pf.sq.cyc, i);
      for (int u = 0; u < NUNITS; u++)
        if (pf.sq.s[u])
          $display("T%0d S c=%0d s=%0d u=%0d r=%0d", SID, pf.sq.cyc, pf.sq.s_slot[u], u,
                   pf.sq.s_rdy[u]);
      if (pf.sq.d)
        $display("T%0d D c=%0d s=%0d pc=%0d op=%02h w1=%08h w2=%08h w3=%08h", SID, pf.sq.cyc,
                 pf.sq.d_slot, pf.sq.d_pc, pf.sq.d_op, pf.sq.d_w1, pf.sq.d_w2, pf.sq.d_w3);
      if (pf.u_mxu)
        $display("T%0d U c=%0d u=1 starve=%0d bp=%0d frz=%0d deny=%0d", SID, pf.sq.cyc,
                 pf.u_mxu_v[0], pf.u_mxu_v[1], pf.u_mxu_v[2], pf.u_mxu_v[3]);
      if (pf.u_q) $display("T%0d U c=%0d u=2 frz=%0d", SID, pf.sq.cyc, pf.u_q_frz);
      if (pf.u_vpu) $display("T%0d U c=%0d u=3 frz=%0d", SID, pf.sq.cyc, pf.u_vpu_frz);
      if (pf.h)
        $display("T%0d H c=%0d bmxu=%0d bdma=%0d amxu=%0d aq=%0d", SID, pf.sq.cyc, pf.h_v[0],
                 pf.h_v[1], pf.h_v[2], pf.h_v[3]);
      if (pf.w) begin
        $display("T%0d P c=%0d n=%0d bm=%0d bd=%0d am=%0d aq=%0d mx=%0d fm=%0d fq=%0d fv=%0d fc=%0d",
                 SID, pf.sq.cyc, pf.w_n, pf.w_p[0], pf.w_p[1], pf.w_p[2], pf.w_p[3], pf.w_p[4],
                 pf.w_p[5], pf.w_p[6], pf.w_p[7], pf.w_p[8]);
        $display("T%0d Q c=%0d n=%0d bs=%0d as=%0d ms=%0d mb=%0d ff=%0d ld=%0d", SID, pf.sq.cyc,
                 pf.w_n, pf.w_q[0], pf.w_q[1], pf.w_q[2], pf.w_q[3], pf.w_q[4], pf.w_q[5]);
      end
    end
`endif
endmodule
