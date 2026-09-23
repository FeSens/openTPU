// One openTPU slice: sequencer (dispatch window + scoreboard), DMA, MXU + ACT RAM, quantizer,
// VPU and TMEM. Units run concurrently. Shared resources are arbitrated every cycle:
//   TMEM   each bank serves 3 reads + 1 write per cycle; units are granted all-or-nothing in
//          the priority order DMA, COLL, MXU drain, QUANT, VPU (a unit that is not granted holds)
//   DRAM B DMA first, then the MXU stream (read responses are routed back by tag)
//   DRAM A QST writes first, then the MXU scale stream
// The slice's DRAM sits outside (otpu_top) so that board wrappers can swap it. The DRAM may
// refuse requests (a_rdy/b_rdy, which must not depend on this cycle's requests), return reads
// after any latency (in order per port), and acknowledge writes late (wr_idle: none pending).
//
// Loader: while the slice is held in reset (rst), ld_start copies ld_n instructions from DRAM
// byte address ld_addr (chunk aligned) into IMEM, one chunk (D/32 instructions) per cycle.
// It runs on sys_rst only, so the board can load a program and then release rst.
module otpu_slice
  import otpu_pkg::*;
#(
  parameter int SID        = 0,
  parameter int S          = 1,
  parameter int D          = 32,
  parameter int MCOLS      = 8,
  parameter int ACT_BLOCKS = 64,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 16,
  parameter int FIFO_DEPTH = 128,
  parameter int LANES      = 8,
  parameter int WIN        = 32,
  parameter int RPB        = 4,       // TMEM reads per bank per cycle
  parameter int WPB        = 2        // TMEM writes per bank per cycle
) (
  input  logic          clk,
  input  logic          sys_rst,
  input  logic          rst,
  // program loader
  input  logic          ld_start,
  input  logic [31:0]   ld_addr,
  input  logic [31:0]   ld_n,
  output logic          ld_busy,
  // DRAM
  input  logic          a_rdy,
  input  logic          b_rdy,
  input  logic          wr_idle,
  output logic          a_req,
  output logic          a_we,
  output logic [31:0]   a_addr,
  output logic [31:0]   a_wdata,
  output logic [3:0]    a_be,
  input  logic          a_rvalid,
  input  logic [31:0]   a_rdata,
  output logic          b_req,
  output logic          b_tag,
  output logic          b_we,
  output logic [D/4-1:0] b_wmask,
  output logic [D*8-1:0] b_wdata,
  output logic [31:0]   b_addr,
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
  output logic [31:0]   icount,
  input  logic          dump
);
  localparam int BW = $clog2(LANES);
  localparam int P_MXU = 0, P_DMA = 1, P_Q = 2, P_VA = 3, P_VB = 4, P_COLL = 5, P_Q2 = 6, P_Q3 = 7, NRP = 8;
  localparam int W_DMA = 0, W_MXU = 1, W_COLL = 2, W_VPU = 3, NWP = 4;
  localparam int G_DMA = 0, G_COLL = 1, G_MXU = 2, G_Q = 3, G_VPU = 4, NG = 5;

  // ---- sequencer
  cmd_t ucmd [NUNITS];
  logic [NUNITS-1:0] ustart, urdy, udone;
  logic urel;
  logic im_we;
  logic [31:0] im_row;
  otpu_seq #(.IMEM_WORDS(IMEM_WORDS), .SID(SID), .S(S), .D(D), .WIN(WIN)) u_seq (
    .clk, .rst, .ucmd, .ustart, .urel, .urdy, .udone, .halted, .error, .icount,
    .im_we, .im_row, .im_data(b_rdata));

  // ---- program loader
  localparam int IPR = D / 32;
  logic [31:0] ld_rows, ld_iss, ld_cmp, ld_a;
  wire ld_req = ld_busy && ld_iss < ld_rows;
  assign im_we  = ld_busy && b_rvalid;
  assign im_row = ld_cmp;
  always_ff @(posedge clk) begin
    if (sys_rst) begin
      ld_busy <= 1'b0;
    end else if (ld_start && !ld_busy) begin
      ld_rows <= (ld_n + IPR - 1) / IPR;
      ld_iss <= '0; ld_cmp <= '0;
      ld_a <= ld_addr >> 2;
      ld_busy <= (ld_n != 0);
    end else if (ld_busy) begin
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

  // ---- TMEM
  logic [NRP-1:0][LANES-1:0]       rq_en, r_en;
  logic [NRP-1:0][LANES-1:0][31:0] r_addr, r_data;
  logic [NWP-1:0][LANES-1:0]       wq_en, w_en;
  logic [NWP-1:0][LANES-1:0][31:0] w_addr, w_data;
  otpu_tmem #(.WORDS(TMEM_WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(WPB), .SID(SID)) u_tmem (
    .clk, .r_en, .r_addr, .r_data, .w_en, .w_addr, .w_data, .dump);
  assign coll_rdata = r_data[P_COLL];

  // ---- ACT RAM
  logic [LANES-1:0]       act_we;
  logic [LANES-1:0][7:0]  act_data;
  logic                   asc_we;
  logic [7:0]             act_row, asc_row;
  logic [31:0]            act_idx, asc_data;
  logic [15:0]            asc_blk, act_rblk;
  logic [MCOLS*D*8-1:0]   act_rdata;
  logic [MCOLS*32-1:0]    act_rscale;
  otpu_actram #(.D(D), .MCOLS(MCOLS), .BLOCKS(ACT_BLOCKS), .LANES(LANES)) u_act (
    .clk, .we(act_we), .w_row(act_row), .w_idx(act_idx), .w_data(act_data), .swe(asc_we),
    .s_row(asc_row), .s_blk(asc_blk), .s_data(asc_data), .r_blk(act_rblk),
    .r_data(act_rdata), .r_scale(act_rscale));

  // ---- units
  logic [NG-1:0] gnt;
  logic        mxu_pop;
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

  otpu_dma #(.D(D), .LANES(LANES)) u_dma (
    .clk, .rst, .start(ustart[U_DMA]), .cmd(ucmd[U_DMA]), .rdy(r_dma), .done(d_dma),
    .b_req(dma_breq), .b_gnt(b_rdy), .b_we(dma_bwe), .b_wmask(dma_bwmask), .b_wdata(dma_bwdata),
    .b_addr(dma_baddr), .b_rvalid(b_rvalid && b_rtag), .b_rdata, .wr_idle,
    .t_ren(rq_en[P_DMA]), .t_raddr(r_addr[P_DMA]), .t_rdata(r_data[P_DMA]),
    .t_wen(wq_en[W_DMA]), .t_waddr(w_addr[W_DMA]), .t_wdata(w_data[W_DMA]));

  otpu_mxu #(.D(D), .MCOLS(MCOLS), .DEPTH(FIFO_DEPTH), .LANES(LANES), .SID(SID)) u_mxu (
    .clk, .rst, .start(ustart[U_MXU]), .go(urel), .cmd(ucmd[U_MXU]), .rdy(r_mxu), .done(d_mxu),
    .computing(mxu_pop),
    .act_blk(act_rblk), .act_data(act_rdata), .act_scale(act_rscale),
    .a_req(mxu_areq), .a_addr(mxu_aaddr), .a_gnt(mxu_agnt), .a_rvalid, .a_rdata,
    .b_req(mxu_breq), .b_addr(mxu_baddr), .b_gnt(mxu_bgnt), .b_rvalid(b_rvalid && !b_rtag),
    .b_rdata,
    .t_ren(rq_en[P_MXU]), .t_raddr(r_addr[P_MXU]), .t_rdata(r_data[P_MXU]),
    .t_wen(wq_en[W_MXU]), .t_waddr(w_addr[W_MXU]), .t_wdata(w_data[W_MXU]), .t_gnt(gnt[G_MXU]));

  otpu_quant #(.D(D), .LANES(LANES), .SID(SID)) u_quant (
    .clk, .rst, .start(ustart[U_Q]), .cmd(ucmd[U_Q]), .rdy(r_q), .done(d_q), .gnt(gnt[G_Q]),
    .t_ren(rq_en[P_Q]), .t_raddr(r_addr[P_Q]), .t_rdata(r_data[P_Q]),
    .t_ren2(rq_en[P_Q2]), .t_raddr2(r_addr[P_Q2]), .t_rdata2(r_data[P_Q2]),
    .t_ren3(q3_en), .t_raddr3(q3_addr), .t_rdata3(r_data[P_Q3][0]),
    .act_we, .act_row, .act_idx, .act_data, .asc_we, .asc_row, .asc_blk, .asc_data,
    .a_want(q_awant), .wr_idle,
    .a_req(q_areq), .a_we(q_awe), .a_addr(q_aaddr), .a_wdata(q_awdata), .a_be(q_abe));

  otpu_vpu #(.LANES(LANES), .SID(SID)) u_vpu (
    .clk, .rst, .start(ustart[U_VPU]), .cmd(ucmd[U_VPU]), .rdy(r_vpu), .done(d_vpu),
    .gnt(gnt[G_VPU]),
    .ta_en(rq_en[P_VA]), .ta_addr(r_addr[P_VA]), .ta_data(r_data[P_VA]),
    .tb_en(rq_en[P_VB]), .tb_addr(r_addr[P_VB]), .tb_data(r_data[P_VB]),
    .tw_en(wq_en[W_VPU]), .tw_addr(w_addr[W_VPU]), .tw_data(w_data[W_VPU]));

  // collective: request from start until acknowledged
  always_ff @(posedge clk) begin
    if (rst) coll_req <= 1'b0;
    else if (ustart[U_COLL]) coll_req <= 1'b1;
    else if (coll_ack) coll_req <= 1'b0;
  end
  assign coll_cmd = ucmd[U_COLL];
  assign rq_en[P_Q3] = LANES'(q3_en);
  always_comb begin
    r_addr[P_Q3] = '0;
    r_addr[P_Q3][0] = q3_addr;
  end
  assign rq_en[P_COLL] = coll_ren;
  assign r_addr[P_COLL] = coll_raddr;
  assign wq_en[W_COLL] = coll_wen;
  assign w_addr[W_COLL] = coll_waddr;
  assign w_data[W_COLL] = coll_wdata;

  assign urdy  = {!coll_req, r_vpu, r_q, r_mxu, r_dma};
  assign udone = {coll_ack, d_vpu, d_q, d_mxu, d_dma};

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

  always_comb begin
    int rc [LANES];
    int wc [LANES];
    int ur [LANES];
    int uw [LANES];
    logic ok;
    logic [NRP-1:0] rp;
    logic [NWP-1:0] wp;
    for (int b = 0; b < LANES; b++) begin rc[b] = 0; wc[b] = 0; end
    coll_gnt_local = 1'b1;
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
      if (g == G_Q && q_awant && !a_rdy) ok = 1'b0;     // QST write the DRAM cannot take
      if (g == G_COLL) begin
        coll_gnt_local = ok;
        gnt[g] = coll_gnt;        // every slice must grant the collective
      end else begin
        gnt[g] = ok;
      end
      if (ok)
        for (int b = 0; b < LANES; b++) begin rc[b] += ur[b]; wc[b] += uw[b]; end
    end
    for (int p = 0; p < NRP; p++) r_en[p] = rq_en[p];
    for (int p = 0; p < NWP; p++) w_en[p] = wq_en[p];
    if (!gnt[G_MXU])  begin r_en[P_MXU] = '0; w_en[W_MXU] = '0; end
    if (!gnt[G_Q])    begin r_en[P_Q] = '0; r_en[P_Q2] = '0; r_en[P_Q3] = '0; end
    if (!gnt[G_VPU])  begin r_en[P_VA] = '0; r_en[P_VB] = '0; w_en[W_VPU] = '0; end
    if (!gnt[G_COLL]) begin r_en[P_COLL] = '0; w_en[W_COLL] = '0; end
  end

  // ---- DRAM ports
  assign mxu_bgnt = !dma_breq && b_rdy;
  assign mxu_agnt = !q_areq && a_rdy;
  always_comb begin
    a_req   = q_areq | mxu_areq;
    a_we    = q_awe;
    a_addr  = q_areq ? q_aaddr : mxu_aaddr;
    a_wdata = q_awdata;
    a_be    = q_abe;
    b_req   = dma_breq | mxu_breq;
    b_tag   = dma_breq;
    b_we    = dma_bwe;
    b_wmask = dma_bwmask;
    b_wdata = dma_bwdata;
    b_addr  = dma_breq ? dma_baddr : mxu_baddr;
    if (ld_busy) begin                // the units are held in reset
      b_req = ld_req; b_tag = 1'b1; b_we = 1'b0; b_addr = ld_a;
    end
  end

`ifndef SYNTHESIS
  // ---- utilisation counters for the profiler: totals at halt, and every `bucket` cycles a
  // P line with the activity of that window (DRAM ports, MXU compute, TMEM-arbitration losses)
  bit trace;
  int bucket;
  initial begin
    trace = $test$plusargs("trace");
    if (!$value$plusargs("bucket=%d", bucket)) bucket = 64;
  end
  longint c_cyc, c_bmxu, c_bdma, c_amxu, c_aq;
  int     w_n, w_bm, w_bd, w_am, w_aq, w_mx, w_fm, w_fq, w_fv, w_fc;
  logic   h_d;
  // a unit "loses" a cycle when it requested TMEM ports and was not granted
  wire lose_mxu = !gnt[G_MXU] && ((|rq_en[P_MXU]) || (|wq_en[W_MXU]));
  wire lose_q   = !gnt[G_Q] && ((|rq_en[P_Q]) || (|rq_en[P_Q2]) || q3_en);
  wire lose_vpu = !gnt[G_VPU] && ((|rq_en[P_VA]) || (|rq_en[P_VB]) || (|wq_en[W_VPU]));
  wire lose_col = !gnt[G_COLL] && ((|rq_en[P_COLL]) || (|wq_en[W_COLL]));
  always_ff @(posedge clk) begin
    if (rst) begin
      c_cyc <= 0; c_bmxu <= 0; c_bdma <= 0; c_amxu <= 0; c_aq <= 0; h_d <= 1'b0;
      w_n <= 0; w_bm <= 0; w_bd <= 0; w_am <= 0; w_aq <= 0; w_mx <= 0;
      w_fm <= 0; w_fq <= 0; w_fv <= 0; w_fc <= 0;
    end else if (!h_d) begin
      c_cyc <= c_cyc + 1;
      if (mxu_breq) c_bmxu <= c_bmxu + 1;
      if (dma_breq) c_bdma <= c_bdma + 1;
      if (mxu_areq) c_amxu <= c_amxu + 1;
      if (q_areq) c_aq <= c_aq + 1;
      h_d <= halted;
      if (trace && halted && !h_d)
        $display("T%0d H c=%0d bmxu=%0d bdma=%0d amxu=%0d aq=%0d", SID, c_cyc, c_bmxu, c_bdma,
                 c_amxu, c_aq);
      if (trace) begin
        if (w_n + 1 == bucket || halted) begin
          $display("T%0d P c=%0d n=%0d bm=%0d bd=%0d am=%0d aq=%0d mx=%0d fm=%0d fq=%0d fv=%0d fc=%0d",
                   SID, c_cyc, w_n + 1, w_bm + mxu_breq, w_bd + dma_breq, w_am + mxu_areq,
                   w_aq + q_areq, w_mx + mxu_pop, w_fm + lose_mxu, w_fq + lose_q,
                   w_fv + lose_vpu, w_fc + lose_col);
          w_n <= 0; w_bm <= 0; w_bd <= 0; w_am <= 0; w_aq <= 0; w_mx <= 0;
          w_fm <= 0; w_fq <= 0; w_fv <= 0; w_fc <= 0;
        end else begin
          w_n <= w_n + 1;
          w_bm <= w_bm + mxu_breq; w_bd <= w_bd + dma_breq; w_am <= w_am + mxu_areq;
          w_aq <= w_aq + q_areq; w_mx <= w_mx + mxu_pop;
          w_fm <= w_fm + lose_mxu; w_fq <= w_fq + lose_q; w_fv <= w_fv + lose_vpu;
          w_fc <= w_fc + lose_col;
        end
      end
    end
  end
`endif
endmodule
