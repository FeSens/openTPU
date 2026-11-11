// EXPERIMENTAL (docs/tmem_nocopy.md): the out-of-context top of the one-copy TMEM. The slice's
// TMEM ports (otpu_slice: 8 read ports, the MXU's with the crossbar; 4 write ports) and the
// slice's arbiter for the TMEM in question, every input registered and every output (the read
// lanes' data, the groups' grants) registered, around one of:
//   VAR 0  otpu_tmem IMPL 0 (a copy per read port, QUANT 3 and COLL guests on the DMA's copy)
//          with the board's write-mask arbiter (otpu_slice gnt_m, ARB_MASK)
//   VAR 3  otpu_tmem IMPL 3 (one copy, NSB RAMs a bank at word-address bit SBP, hashed by SBH,
//          request-priority addresses) with otpu_slice's TMEM_IMPL 3 arbiter (gnt_s: tm_bx /
//          tm_bs, the split reads tm_pre / tm_sv)
//   VAR 4  otpu_tmem IMPL 3 RPL (replays) with the board's arbiter and the replays' holds
//          (otpu_slice TMEM_RPL: tm_frz into the grants, the VPU's registered enable from
//          tm_lost, as otpu_vpu's WBUF en_r)
//   VAR 5  VAR 4 with true dual-port RAMs (otpu_slice TMEM_TDP: the MXU, the DMA, QUANT and VPU A
//          on port B; VPU B, QUANT 3, QUANT 2 and COLL on port A with the writes)
//   VAR 7  VAR 5 with the RAMs pumped at 2x (otpu_tmem PMP 1: a write takes no read port, no
//          bypass; clk_f from the MMCM, as VAR 6)
//   VAR 6  P3: the copies pumped at 2x (rtl/exp/tmem_pump/otpu_tmem_pump IMPL 0, three copies)
//          with the board's arbiter, as VAR 0; the clocks from an MMCM (core and 2x core)
// MMCM 1 (VAR 6 always): the core clock from an MMCM on clk (CLKIN_NS; FAMILY 0 7-series, 1
// UltraScale+), as tmem_pump_top, so P3 and the others compare on the same clocking.
// So the paths it times are the TMEM's own and the grant's up to the units' enables (the units'
// requests come from registers here; in the slice they come out of each unit's state).
module tmem_nc_top #(
  parameter int VAR = 3,
  parameter int WORDS = 1 << 16,
  parameter int NSB = 8,
  parameter int SBP = 3,
  parameter int SBH = 0,
  parameter bit GST = 1'b0,          // VAR 4 / 5: QUANT 3 and COLL the DMA's guests (TMEM_GST)
  parameter int SDM = -1,            // VAR 5 / 7: the read ports' RAM sides (otpu_slice TMEM_SDM)
  parameter int PRK = 0,             // VAR 4 / 5 / 7: the read ports' ranks (otpu_slice TMEM_PRK)
  parameter bit MMCM = 1'b0,
  parameter int FAMILY = 0,
  parameter real CLKIN_NS = 7.5
) (
  input  logic                          clk,
  input  logic [7:0][7:0]               rq_in,
  input  logic [7:0][7:0][31:0]         ra_in,
  input  logic [3:0][7:0]               wq_in,
  input  logic [3:0][7:0][31:0]         wa_in,
  input  logic [3:0][7:0][31:0]         wd_in,
  input  logic                          coll_gnt_in,
  input  logic                          v_room_in,      // VAR 4: the VPU's write buffer room
  output logic [7:0][7:0][31:0]         r_data_out,
  output logic [4:0]                    gnt_out
);
  localparam int LANES = 8, NRP = 8, NWP = 4, NG = 5, BW = 3;
  localparam int P_MXU = 0, P_DMA = 1, P_Q = 2, P_VA = 3, P_VB = 4, P_Q3 = 5, P_Q2 = 6, P_COLL = 7;
  localparam int W_DMA = 0, W_MXU = 1, W_COLL = 2, W_VPU = 3;
  localparam int G_DMA = 0, G_COLL = 1, G_VPU = 2, G_MXU = 3, G_Q = 4;
  localparam logic [NRP-1:0] GEN_R = NRP'(1) << P_MXU;
  localparam logic [NWP-1:0] GEN_W = NWP'(1) << W_MXU;
  localparam logic [NRP-1:0] SH_MASK = (NRP'(1) << P_Q3) | (NRP'(1) << P_COLL);
  // the read ports' RAM sides as otpu_slice's TM_SIDED (pumped, VAR 7: QUANT and VPU A on port A)
  localparam logic [NRP-1:0] TM_SIDED = (VAR == 7) ? ((NRP'(1) << P_Q) | (NRP'(1) << P_VA)) :
                                        ((NRP'(1) << P_VB) | (NRP'(1) << P_Q3) | (NRP'(1) << P_Q2) |
                                         (NRP'(1) << P_COLL));
  localparam logic [NRP-1:0] TM_SIDE = (SDM >= 0) ? NRP'(SDM) : TM_SIDED;
  localparam logic [NRP-1:0][2:0] PRANK3D = {3'd1, 3'd6, 3'd7, 3'd3, 3'd2, 3'd5, 3'd0, 3'd4};
  localparam logic [NRP-1:0][2:0] PRANK3 = (PRK != 0) ? (3 * NRP)'(PRK) : PRANK3D;
  localparam logic [NRP-1:0][2:0] RGRP = {3'(G_COLL), 3'(G_Q), 3'(G_Q), 3'(G_VPU), 3'(G_VPU),
                                          3'(G_Q), 3'(G_DMA), 3'(G_MXU)};
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

  logic ck, ck_f;                  // the core clock and (VAR 6) the 2x clock
  if (MMCM || VAR == 6 || VAR == 7) begin : g_mmcm
    logic clkfb, ck_u, ckf_u;
    if (FAMILY == 0) begin : g_k7
      MMCME2_ADV #(
        .BANDWIDTH("OPTIMIZED"), .COMPENSATION("INTERNAL"), .DIVCLK_DIVIDE(1),
        .CLKFBOUT_MULT_F(8.0), .CLKFBOUT_PHASE(0.0), .CLKIN1_PERIOD(CLKIN_NS),
        .CLKOUT0_DIVIDE_F(8.0), .CLKOUT0_PHASE(0.0), .CLKOUT0_DUTY_CYCLE(0.5),
        .CLKOUT1_DIVIDE(4), .CLKOUT1_PHASE(0.0), .CLKOUT1_DUTY_CYCLE(0.5)
      ) u_mmcm (
        .CLKIN1(clk), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(clkfb), .CLKFBOUT(clkfb),
        .CLKFBOUTB(), .CLKOUT0(ck_u), .CLKOUT0B(), .CLKOUT1(ckf_u), .CLKOUT1B(),
        .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
        .LOCKED(), .PWRDWN(1'b0), .RST(1'b0),
        .DADDR(7'd0), .DCLK(1'b0), .DEN(1'b0), .DI(16'd0), .DO(), .DRDY(), .DWE(1'b0),
        .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(), .CLKINSTOPPED(), .CLKFBSTOPPED());
      BUFG u_bs (.I(ck_u), .O(ck));
      BUFG u_bf (.I(ckf_u), .O(ck_f));
    end else begin : g_us
      MMCME4_ADV #(
        .BANDWIDTH("OPTIMIZED"), .COMPENSATION("INTERNAL"), .DIVCLK_DIVIDE(1),
        .CLKFBOUT_MULT_F(8.0), .CLKFBOUT_PHASE(0.0), .CLKIN1_PERIOD(CLKIN_NS),
        .CLKOUT0_DIVIDE_F(8.0), .CLKOUT0_PHASE(0.0), .CLKOUT0_DUTY_CYCLE(0.5),
        .CLKOUT1_DIVIDE(4), .CLKOUT1_PHASE(0.0), .CLKOUT1_DUTY_CYCLE(0.5)
      ) u_mmcm (
        .CLKIN1(clk), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(clkfb), .CLKFBOUT(clkfb),
        .CLKFBOUTB(), .CLKOUT0(ck_u), .CLKOUT0B(), .CLKOUT1(ckf_u), .CLKOUT1B(),
        .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
        .LOCKED(), .PWRDWN(1'b0), .RST(1'b0), .CDDCREQ(1'b0), .CDDCDONE(),
        .DADDR(7'd0), .DCLK(1'b0), .DEN(1'b0), .DI(16'd0), .DO(), .DRDY(), .DWE(1'b0),
        .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .PSDONE(), .CLKINSTOPPED(), .CLKFBSTOPPED());
      BUFGCE u_bs (.I(ck_u), .CE(1'b1), .O(ck));
      BUFGCE u_bf (.I(ckf_u), .CE(1'b1), .O(ck_f));
    end
  end else begin : g_ck
    assign ck = clk;
    assign ck_f = 1'b0;
  end

  logic [NRP-1:0][LANES-1:0]       rq_en, r_en, rq_t;
  logic [NRP-1:0][LANES-1:0][31:0] r_addr, r_data;
  logic [NWP-1:0][LANES-1:0]       wq_en, w_en;
  logic [NWP-1:0]                  w_gnt;
  logic [NWP-1:0][LANES-1:0][31:0] w_addr, w_data;
  logic                            coll_gnt, v_room;
  logic [NG-1:0]                   gnt;
  always_ff @(posedge ck) begin
    rq_en <= rq_in; r_addr <= ra_in; wq_en <= wq_in; w_addr <= wa_in; w_data <= wd_in;
    coll_gnt <= coll_gnt_in; v_room <= v_room_in;
    r_data_out <= r_data;
    gnt_out <= gnt;
  end

  // the write banks' pairwise conflicts (otpu_slice amk / acf)
  logic [NG-1:0][LANES-1:0] amk;
  logic [NG-1:0][NG-1:0]    acf;
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

  logic [NRP-1:0] tm_bx, tm_bs, tm_sv, tm_pre, tm_lost, tm_frz;
  logic           v_en;            // VAR 4: the VPU's registered enable (otpu_vpu en_r)
  if (VAR == 4 || VAR == 5 || VAR == 7) begin : g_v4
    logic [NG-1:0] frzg;
    always_comb
      for (int g = 0; g < NG; g++)
        frzg[g] = g != G_VPU && g != G_COLL && (tm_frz & grp_rports(g)) != '0;
    always_comb begin
      logic [NG-1:0] aok;
      for (int g = 0; g < NG; g++) begin
        aok[g] = !frzg[g] && !(g == G_COLL && tm_bx[P_COLL] && |rq_en[P_COLL]);
        if (GST && ((g == G_Q && (|rq_en[P_Q3]) && (|rq_en[P_DMA])) ||
                    (g == G_COLL && (|rq_en[P_COLL]) && ((|rq_en[P_DMA]) || (|rq_en[P_Q3])))))
          aok[g] = 1'b0;
        for (int h = 0; h < g; h++) if (aok[h] && acf[g][h]) aok[g] = 1'b0;
      end
      gnt = aok;
      gnt[G_COLL] = coll_gnt;
    end
    always_comb begin
      logic [NRP-1:0] off;
      off = '0;
      for (int g = 0; g < NG; g++)
        if ((g == G_VPU) ? !v_en : frzg[g]) off = off | grp_rports(g);
      // GST: the collective asks only when the guests' rule lets it read (otpu_slice coll_blk)
      if (GST && (|rq_en[P_COLL]) && ((|rq_en[P_DMA]) || (|rq_en[P_Q3]))) off[P_COLL] = 1'b1;
      for (int p = 0; p < NRP; p++) rq_t[p] = off[p] ? '0 : rq_en[p];
    end
    always_ff @(posedge ck) v_en <= v_room && !(tm_lost[P_VA] || tm_lost[P_VB]);
    assign tm_sv = '0;
    assign tm_pre = '0;
  end else if (VAR == 0 || VAR == 6) begin : g_v0
    wire q3_blk = (|rq_en[P_Q3]) && (|rq_en[P_DMA]);
    wire coll_blk = (|rq_en[P_COLL]) && ((|rq_en[P_DMA]) || (|rq_en[P_Q3]));
    always_comb begin
      logic [NG-1:0] aok;
      for (int g = 0; g < NG; g++) begin
        aok[g] = 1'b1;
        for (int h = 0; h < g; h++) if (aok[h] && acf[g][h]) aok[g] = 1'b0;
        if ((g == G_Q && q3_blk) || (g == G_COLL && coll_blk)) aok[g] = 1'b0;
      end
      gnt = aok;
      gnt[G_COLL] = coll_gnt;
    end
    assign rq_t = rq_en;
    assign tm_sv = '0;
    assign tm_pre = '0;
    assign v_en = 1'b1;
  end else begin : g_v3
    assign v_en = 1'b1;
    always_comb
      for (int p = 0; p < NRP; p++) rq_t[p] = tm_sv[p] ? '0 : rq_en[p];
    always_comb begin
      logic [NG-1:0]  gs;
      logic [NRP-1:0] rp, ask;
      logic           rd_ok, self;
      tm_pre = '0;
      for (int p = 0; p < NRP; p++) ask[p] = |rq_t[p];
      for (int g = 0; g < NG; g++) begin
        rp = grp_rports(g);
        rd_ok = (tm_bx & rp & ask) == '0;
        self = (tm_bs & rp & ask) != '0;
        gs[g] = rd_ok && !self;
        for (int h = 0; h < g; h++) if (gs[h] && acf[g][h]) gs[g] = 1'b0;
        if (rd_ok && self) tm_pre = tm_pre | (rp & ask & ~tm_bs);
      end
      gnt = gs;
      gnt[G_COLL] = coll_gnt;
    end
    always_ff @(posedge ck)
      for (int g = 0; g < NG; g++) begin
        logic [NRP-1:0] rp;
        rp = grp_rports(g);
        for (int p = 0; p < NRP; p++)
          if (rp[p]) tm_sv[p] <= gnt[g] ? 1'b0 : (tm_sv[p] | tm_pre[p]);
      end
  end

  // the units' enables (otpu_slice; the VPU's reads by its grant: WBUF 0 outside ARB_MASK)
  always_comb begin
    for (int p = 0; p < NRP; p++) r_en[p] = rq_en[p];
    for (int p = 0; p < NWP; p++) w_en[p] = wq_en[p];
    w_gnt = '1;
    w_gnt[W_MXU] = gnt[G_MXU];
    w_gnt[W_VPU] = gnt[G_VPU];
    w_gnt[W_COLL] = gnt[G_COLL];
    if (!gnt[G_MXU])  begin r_en[P_MXU] = '0; w_en[W_MXU] = '0; end
    if (!gnt[G_Q])    begin r_en[P_Q] = '0; r_en[P_Q2] = '0; r_en[P_Q3] = '0; end
    if (VAR == 3 && !gnt[G_VPU]) begin r_en[P_VA] = '0; r_en[P_VB] = '0; end
    if ((VAR == 4 || VAR == 5 || VAR == 7) && !v_en) begin r_en[P_VA] = '0; r_en[P_VB] = '0; end
    if (!gnt[G_VPU])  w_en[W_VPU] = '0;
    if (!gnt[G_COLL]) begin r_en[P_COLL] = '0; w_en[W_COLL] = '0; end
    if (VAR == 3)
      for (int p = 0; p < NRP; p++) begin
        if (tm_sv[p]) r_en[p] = '0;
        if (tm_pre[p]) r_en[p] = rq_en[p];
      end
  end

  if (VAR == 6) begin : g_p3
    otpu_tmem_pump #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .GEN_R(GEN_R),
                     .GEN_W(GEN_W), .SH_HOST(P_DMA), .SH_MASK(SH_MASK), .IMPL(0)) u_tmem (
      .clk(ck), .clk_f(ck_f), .r_en, .r_req(rq_t), .r_addr, .r_ben('0), .r_bad('0), .r_data, .w_en, .w_req(wq_en), .w_gnt,
      .w_addr, .w_data);
    assign tm_bx = '0;
    assign tm_bs = '0;
    assign tm_lost = '0;
    assign tm_frz = '0;
  end else begin : g_tm
    otpu_tmem #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(1), .GEN_R(GEN_R),
                .GEN_W(GEN_W), .SH_HOST(P_DMA), .SH_MASK((VAR == 0 || GST) ? SH_MASK : '0), .SID(0),
                .IMPL(VAR == 0 ? 0 : 3), .PRANK(PRANK3), .NSB(NSB), .SBP(SBP), .GRP(RGRP),
                .SBH(SBH), .RPL(VAR == 4 || VAR == 5 || VAR == 7), .RPLM(~(NRP'(1) << P_COLL)), .TOPP(P_DMA),
                .TDP(VAR == 5 || VAR == 7), .SIDE(TM_SIDE), .GST(GST), .PMP(VAR == 7 ? 1 : 0)) u_tmem (
      .clk(ck), .clk_f(ck_f), .rst(1'b0), .r_en, .r_req(rq_t), .r_addr, .r_ben('0), .r_bad('0), .r_data, .w_en, .w_req(wq_en), .w_gnt, .w_addr,
      .w_data, .blk_x(tm_bx), .blk_s(tm_bs), .lost(tm_lost), .frz(tm_frz), .dump(1'b0));
  end
endmodule
