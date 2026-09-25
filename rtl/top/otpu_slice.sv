// One openTPU slice: sequencer (dispatch window + scoreboard), DMA, MXU + ACT RAM, quantizer,
// VPU and TMEM. Units run concurrently. Shared resources are arbitrated every cycle:
//   TMEM   each bank serves 3 reads + 1 write per cycle; units are granted all-or-nothing in
//          the priority order DMA, COLL, MXU drain, QUANT, VPU (a unit that is not granted holds)
//   DRAM B DMA first, then the MXU stream (read responses are routed back by tag)
//   DRAM A the MXU scale stream (reads); QST writes have their own scalar write port (SW)
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
  parameter int WPB        = 2,       // TMEM writes per bank per cycle
  parameter int MXU_IMPL   = 0,       // MXU dot product: 0 adder tree, 1 DSP cascade chains
  parameter int MXU_CL     = 16,      // cascade chain length
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,  // VPU lanes with the composite functions
  parameter int PQ_WIN     = 64       // cycles per P/Q counter window (+bucket= in simulation)
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
  input  logic          sw_rdy,
  input  logic          wr_idle,
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
  output perf_t         pf,         // activity and trace events, a cycle late (otpu_pkg)
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
  seq_ev_t sq_ev;
  otpu_seq #(.IMEM_WORDS(IMEM_WORDS), .SID(SID), .S(S), .D(D), .WIN(WIN)) u_seq (
    .clk, .rst, .ucmd, .ustart, .urel, .urdy, .udone, .halted, .error, .icount, .ev(sq_ev),
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
  logic [NWP-1:0]                  w_gnt;   // each write port's grant (the DMA always writes)
  // the MXU's ports keep the crossbar (its drain writes scattered rows); the others are rotators
  otpu_tmem #(.WORDS(TMEM_WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(WPB),
              .GEN_R(NRP'(1) << P_MXU), .GEN_W(NWP'(1) << W_MXU), .SID(SID)) u_tmem (
    .clk, .r_en, .r_req(rq_en), .r_addr, .r_data, .w_en, .w_req(wq_en), .w_gnt, .w_addr,
    .w_data, .dump);
  assign coll_rdata = r_data[P_COLL];
  // the units' TMEM requests
  logic [LANES-1:0]        dma_ren, dma_wen, mxu_ren, mxu_wen, q_ren, q_ren2;
  logic [LANES-1:0]        va_ren, vb_ren, v_wen;
  logic [LANES-1:0][31:0]  dma_raddr, dma_waddr, dma_wdata, mxu_raddr, mxu_waddr, mxu_wdata;
  logic [LANES-1:0][31:0]  q_raddr, q_raddr2, va_raddr, vb_raddr, v_waddr, v_wdata;

  // ---- ACT RAM
  logic [LANES-1:0]       act_we;
  logic [LANES-1:0][7:0]  act_data;
  logic                   asc_we;
  logic [7:0]             act_row, asc_row;
  logic [31:0]            act_idx, asc_data;
  logic [15:0]            asc_blk, act_rblk;
  logic                   act_ren;
  logic [MCOLS*D*8-1:0]   act_rdata;
  logic [MCOLS*32-1:0]    act_rscale;
  otpu_actram #(.D(D), .MCOLS(MCOLS), .BLOCKS(ACT_BLOCKS), .LANES(LANES)) u_act (
    .clk, .we(act_we), .w_row(act_row), .w_idx(act_idx), .w_data(act_data), .swe(asc_we),
    .s_row(asc_row), .s_blk(asc_blk), .s_data(asc_data), .ren(act_ren), .r_blk(act_rblk),
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
  localparam int FW = $clog2(FIFO_DEPTH) + 1;
  logic [FW-1:0] mxu_level;
  logic mxu_starve, mxu_block, mxu_u, q_u, v_u;
  logic [3:0][31:0] mxu_uv;
  logic [31:0] q_frz, v_frz;

  otpu_dma #(.D(D), .LANES(LANES)) u_dma (
    .clk, .rst, .start(ustart[U_DMA]), .cmd(ucmd[U_DMA]), .rdy(r_dma), .done(d_dma),
    .b_req(dma_breq), .b_gnt(b_rdy), .b_we(dma_bwe), .b_wmask(dma_bwmask), .b_wdata(dma_bwdata),
    .b_addr(dma_baddr), .b_rvalid(b_rvalid && b_rtag), .b_rdata, .wr_idle,
    .t_ren(dma_ren), .t_raddr(dma_raddr), .t_rdata(r_data[P_DMA]),
    .t_wen(dma_wen), .t_waddr(dma_waddr), .t_wdata(dma_wdata));

  otpu_mxu #(.D(D), .MCOLS(MCOLS), .DEPTH(FIFO_DEPTH), .LANES(LANES), .IMPL(MXU_IMPL),
             .CL(MXU_CL), .SID(SID)) u_mxu (
    .clk, .rst, .start(ustart[U_MXU]), .go(urel), .cmd(ucmd[U_MXU]), .rdy(r_mxu), .done(d_mxu),
    .computing(mxu_pop), .pf_level(mxu_level), .pf_starve(mxu_starve), .pf_block(mxu_block),
    .pf_u(mxu_u), .pf_uv(mxu_uv),
    .act_blk(act_rblk), .act_ren, .act_data(act_rdata), .act_scale(act_rscale),
    .a_req(mxu_areq), .a_addr(mxu_aaddr), .a_gnt(mxu_agnt), .a_rvalid, .a_rdata,
    .b_req(mxu_breq), .b_addr(mxu_baddr), .b_gnt(mxu_bgnt), .b_rvalid(b_rvalid && !b_rtag),
    .b_rdata,
    .t_ren(mxu_ren), .t_raddr(mxu_raddr), .t_rdata(r_data[P_MXU]),
    .t_wen(mxu_wen), .t_waddr(mxu_waddr), .t_wdata(mxu_wdata), .t_gnt(gnt[G_MXU]));

  otpu_quant #(.D(D), .LANES(LANES), .SID(SID)) u_quant (
    .clk, .rst, .start(ustart[U_Q]), .cmd(ucmd[U_Q]), .rdy(r_q), .done(d_q), .gnt(gnt[G_Q]),
    .t_ren(q_ren), .t_raddr(q_raddr), .t_rdata(r_data[P_Q]),
    .t_ren2(q_ren2), .t_raddr2(q_raddr2), .t_rdata2(r_data[P_Q2]),
    .t_ren3(q3_en), .t_raddr3(q3_addr), .t_rdata3(r_data[P_Q3][0]),
    .act_we, .act_row, .act_idx, .act_data, .asc_we, .asc_row, .asc_blk, .asc_data,
    .a_want(q_awant), .wr_idle,
    .a_req(q_areq), .a_we(q_awe), .a_addr(q_aaddr), .a_wdata(q_awdata), .a_be(q_abe),
    .pf_u(q_u), .pf_frz(q_frz));

  otpu_vpu #(.LANES(LANES), .CL(VPU_CL), .SID(SID)) u_vpu (
    .clk, .rst, .start(ustart[U_VPU]), .cmd(ucmd[U_VPU]), .rdy(r_vpu), .done(d_vpu),
    .gnt(gnt[G_VPU]),
    .ta_en(va_ren), .ta_addr(va_raddr), .ta_data(r_data[P_VA]),
    .tb_en(vb_ren), .tb_addr(vb_raddr), .tb_data(r_data[P_VB]),
    .tw_en(v_wen), .tw_addr(v_waddr), .tw_data(v_wdata), .pf_u(v_u), .pf_frz(v_frz));

  // collective: request from start until acknowledged
  always_ff @(posedge clk) begin
    if (rst) coll_req <= 1'b0;
    else if (ustart[U_COLL]) coll_req <= 1'b1;
    else if (coll_ack) coll_req <= 1'b0;
  end
  assign coll_cmd = ucmd[U_COLL];
  // the TMEM request arrays, each assembled in one process from the units' ports
  always_comb begin
    rq_en = '0; r_addr = '0; wq_en = '0; w_addr = '0; w_data = '0;
    rq_en[P_DMA] = dma_ren;   r_addr[P_DMA] = dma_raddr;
    rq_en[P_MXU] = mxu_ren;   r_addr[P_MXU] = mxu_raddr;
    rq_en[P_Q] = q_ren;       r_addr[P_Q] = q_raddr;
    rq_en[P_Q2] = q_ren2;     r_addr[P_Q2] = q_raddr2;
    rq_en[P_Q3][0] = q3_en;   r_addr[P_Q3][0] = q3_addr;
    rq_en[P_VA] = va_ren;     r_addr[P_VA] = va_raddr;
    rq_en[P_VB] = vb_ren;     r_addr[P_VB] = vb_raddr;
    rq_en[P_COLL] = coll_ren; r_addr[P_COLL] = coll_raddr;
    wq_en[W_DMA] = dma_wen;   w_addr[W_DMA] = dma_waddr;   w_data[W_DMA] = dma_wdata;
    wq_en[W_MXU] = mxu_wen;   w_addr[W_MXU] = mxu_waddr;   w_data[W_MXU] = mxu_wdata;
    wq_en[W_VPU] = v_wen;     w_addr[W_VPU] = v_waddr;     w_data[W_VPU] = v_wdata;
    wq_en[W_COLL] = coll_wen; w_addr[W_COLL] = coll_waddr; w_data[W_COLL] = coll_wdata;
  end

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

  // The board build (TMEM replicated per read port, RPB >= NRP * LANES: reads never conflict;
  // WPB = 1) only needs write-bank masks: a unit is granted when no bank it writes was taken by
  // a higher-priority unit this cycle (a unit's own lanes write distinct banks). Shallow logic,
  // no counters. Other configurations count reads and writes per bank.
  localparam bit ARB_MASK = (RPB >= NRP * LANES) && (WPB == 1);
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
  always_comb begin
    for (int g = 0; g < NG; g++) begin
      aok[g] = 1'b1;
      for (int h = 0; h < g; h++) if (aok[h] && acf[g][h]) aok[g] = 1'b0;
      if (g == G_Q && q_awant && !sw_rdy) aok[g] = 1'b0;   // QST write the DRAM cannot take
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
    if (!gnt[G_VPU])  begin r_en[P_VA] = '0; r_en[P_VB] = '0; w_en[W_VPU] = '0; end
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
    if (ld_busy) begin                // the units are held in reset
      b_req = ld_req; b_tag = 1'b1; b_we = 1'b0; b_addr = ld_a;
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
