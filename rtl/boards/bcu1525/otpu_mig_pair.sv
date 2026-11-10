// One DDR4 DIMM's NC virtual channels (the BCU1525 at WG 8, targets/bcu1525-wg.toml: two; at two
// slices, targets/bcu1525-s2.toml: four, a slice's 8 channels on its two DIMMs): the AXI sides of NC
// channel bridges (otpu_hbm_ch, AXI3 256-bit at aclk, no IDs, PC 0: channel beat m at byte m x CB
// of the channel's 2^CL bytes) onto one AXI4 port of 512 bits (aclk; the DIMM's MIG behind a
// SmartConnect that also takes the host's DMA and crosses into the MIG's ui_clk). At 300 MHz a
// bridge's 256-bit port moves 9.6 GB/s, two together the DIMM's 19.2 (DDR4-2400); four leave each
// bridge half its port.
//
// Address: channel s's byte o (o < 2^CL) is the DIMM's {o[CL-1:12], s, o[11:0]} (CL + log2 NC
// bits): the channels interleave at 4 KiB inside the DIMM, so their streams, which move together (a
// group step reads one beat of every channel), share the DIMM's rows (the MIG's ROW_COLUMN_BANK map
// puts the bank bits below bit 12), and a burst, which never crosses 4 KiB, stays whole. The
// host's DMA takes the same permutation (otpu_fpga_top_bcu_wg's remap: channel j at j x 2^CL of
// the card). CL 29 (512 MiB a channel, 1 GiB a DIMM) was the first build's; 32 (4 GiB, all of an
// 8 GB DIMM) the 32 GB card's; 33 needs 16 GB DIMMs (the MIG's AXI at 34 bits).
//
// BHW (a bank-half window; 0: none): a channel's first 2^BHW bytes (a slice's first 2^(BHW + 3):
// at 30 its first 8 GiB) take the MIG's bank-high bit (BHB: WG_MIGMAP b15's 16, the DIMM's address
// bit) from the window's top bit, and that bit's place takes the bank bit: the window's lower half
// (at 30 the slice's first 4 GiB: what a decode writes, the DeltaNet states among them) on the
// banks with that bit 0, its upper half (the DeltaNet mixers' weights) on the others, every byte
// above the window on the map as it is (all banks). A swap of two bits decided by bits neither
// touches: its own inverse, so a bijection; the host's remap (otpu_fpga_top_bcu_wg) and the board
// model's DIMM (otpu_dimm_model) take the same (docs/bcu1525_wg.md, the bank-half window). At
// NC 4 the window's top bit sits a place higher in the DIMM's address (the channel's two bits).
//
// AR and AW go round the bridges that ask (one burst a turn, from a register, as otpu_axi3_mux2:
// at NC 2 they alternate), the burst's 256-bit beats (an even count: whole 128-byte channel beats)
// as half as many 512-bit beats, ID {s, 0...} (s in the top log2 NC bits). R: the 512-bit beats go
// by RID's top bits into the bridge's FIFO (g_r's lanes, RD entries, held back only when it is full) and leave as two 256-bit beats
// each, the low half first (the lower address), RLAST on the last's high half: one bridge drains
// a 512-bit beat every two cycles, so the DIMM's R stream at a beat a cycle keeps two busy when
// their bursts alternate. W follows AW's order (AXI4 has no WID): each granted AW queues its
// bridge in u_wo, and W takes two 256-bit beats of the head's bridge into one 512-bit beat (the
// first held in a register, the second passed with it), a 512-bit beat every two cycles (a bridge
// sends its burst's W only with all its data in: otpu_hbm_ch). B goes back by BID's top bits.
// WOC > 1: u_wo's head in WOC copies (otpu_sfifo HCP), each picking 288 / WOC of the W beat's data
// and strobe bits (B1r: u_wo's head into the 288 bits' muxes and on into the host mux's W ring,
// -0.39 ns at 300 MHz, ~97 % route); 1, the default: the head alone. RS: R through a register a
// bridge (its beat, RLAST and RRESP every cycle, KEEP, no enable; valid the accepted beat's) into
// its FIFO a cycle later, the FIFO's fill counted from the accept (credits: the beat on its way
// holds its entry), so RREADY drops a beat early and the FIFO never refuses the stage (B1r: the
// host mux's R ring into the four bridges' FIFO lanes, -0.39 ns at 300 MHz, ~97 % route; one
// cycle more of R latency); 0, the default: R straight into the FIFOs, RREADY their not-full.
module otpu_mig_pair #(
  parameter int IDW = 4,                   // the AXI4 port's IDs
  parameter int RD  = 32,                  // each bridge's R FIFO, 512-bit beats
  parameter int CL  = 29,                  // a channel's byte address bits (the DIMM's CL + log2 NC)
  parameter int NC  = 2,                   // the DIMM's channels (2 or 4)
  parameter int BHW = 0,                   // the bank-half window's channel bytes, log2 (0: none)
  parameter int BHB = 16,                  // the MIG map's bank-high bit (a DIMM address bit)
  parameter int WOC = 1,                   // u_wo's head in copies (1, 2, 4 or 8)
  parameter bit RS  = 1'b0                 // R through a register a bridge, credits
) (
  input  logic              clk,           // aclk
  input  logic              rst,
  // the bridges (otpu_hbm_ch m_*), AXI3 256-bit
  input  logic [NC-1:0]        s_awvalid,
  output logic [NC-1:0]        s_awready,
  input  logic [NC-1:0][33:0]  s_awaddr,
  input  logic [NC-1:0][3:0]   s_awlen,
  input  logic [NC-1:0]        s_wvalid,
  output logic [NC-1:0]        s_wready,
  input  logic [NC-1:0][255:0] s_wdata,
  input  logic [NC-1:0][31:0]  s_wstrb,
  input  logic [NC-1:0]        s_wlast,
  output logic [NC-1:0]        s_bvalid,
  input  logic [NC-1:0]        s_bready,
  output logic [1:0]           s_bresp,    // every bridge's (one valid at a time)
  input  logic [NC-1:0]        s_arvalid,
  output logic [NC-1:0]        s_arready,
  input  logic [NC-1:0][33:0]  s_araddr,
  input  logic [NC-1:0][3:0]   s_arlen,
  output logic [NC-1:0]        s_rvalid,
  input  logic [NC-1:0]        s_rready,
  output logic [NC-1:0][255:0] s_rdata,
  output logic [NC-1:0][1:0]   s_rresp,
  output logic [NC-1:0]        s_rlast,
  // the DIMM's AXI4 port, 512-bit (AxSIZE 64 bytes, INCR: the top ties them)
  output logic              m_awvalid,
  input  logic              m_awready,
  output logic [IDW-1:0]    m_awid,
  output logic [CL+$clog2(NC)-1:0] m_awaddr,
  output logic [7:0]        m_awlen,
  output logic              m_wvalid,
  input  logic              m_wready,
  output logic [511:0]      m_wdata,
  output logic [63:0]       m_wstrb,
  output logic              m_wlast,
  input  logic              m_bvalid,
  output logic              m_bready,
  input  logic [IDW-1:0]    m_bid,
  input  logic [1:0]        m_bresp,
  output logic              m_arvalid,
  input  logic              m_arready,
  output logic [IDW-1:0]    m_arid,
  output logic [CL+$clog2(NC)-1:0] m_araddr,
  output logic [7:0]        m_arlen,
  input  logic              m_rvalid,
  output logic              m_rready,
  input  logic [IDW-1:0]    m_rid,
  input  logic [511:0]      m_rdata,
  input  logic [1:0]        m_rresp,
  input  logic              m_rlast
);
  localparam int LNC = $clog2(NC);         // (the channel's bits in the DIMM's address)
  localparam int WB = BHW != 0 ? BHW : 14, HB = BHW != 0 ? BHB : 13;    // (BHW 0: indices in range)
  initial if (CL < 13 || CL + LNC > 34 || (NC != 2 && NC != 4) || IDW <= LNC)
    $fatal(1, "otpu_mig_pair: CL %0d, NC %0d, IDW %0d (NC 2 or 4, CL 13 .. 34 - log2 NC, IDW > log2 NC)",
           CL, NC, IDW);
  initial if (WOC != 1 && WOC != 2 && WOC != 4 && WOC != 8)
    $fatal(1, "otpu_mig_pair: WOC %0d (1, 2, 4 or 8)", WOC);
  initial if (BHW != 0 && (BHW >= CL || BHB < 12 + LNC || BHB >= BHW - 1 + LNC))
    $fatal(1, "otpu_mig_pair: BHW %0d, BHB %0d (12 + log2 NC <= BHB < BHW - 1 + log2 NC, BHW < CL)",
           BHW, BHB);

  // the DIMM's byte of channel s's byte a (a channel of 2^CL bytes)
  function automatic logic [CL+LNC-1:0] dimm(input logic [33:0] a, input logic [LNC-1:0] s);
    return {a[CL-1:12], s, a[11:0]};
  endfunction
  // the next bridge after `last` that asks (round robin; `last` itself the last choice)
  function automatic logic [LNC-1:0] rr(input logic [NC-1:0] req, input logic [LNC-1:0] last);
    logic [LNC-1:0] k;
    rr = last;
    for (int i = NC; i >= 1; i--) begin
      k = LNC'(int'(last) + i);
      if (req[k]) rr = k;
    end
  endfunction

  // a DIMM byte d through the bank-half window: inside it (its channel bytes below 2^BHW) the
  // bank-high bit and the window's top bit trade places
  function automatic logic [CL+LNC-1:0] bhw(input logic [CL+LNC-1:0] d);
    logic [CL+LNC-1:0] e;
    e = d;
    if (BHW != 0 && d[CL+LNC-1:WB+LNC] == '0) begin
      e[HB] = d[WB-1+LNC];
      e[WB-1+LNC] = d[HB];
    end
    return e;
  endfunction

  // ---- AR: a register, loaded from the bridge whose turn it is (or another, if only it asks)
  logic [LNC-1:0] ar_last, ar_pick;
  assign ar_pick = rr(s_arvalid, ar_last);
  for (genvar s = 0; s < NC; s++) begin : g_arr
    assign s_arready[s] = (!m_arvalid || m_arready) && s_arvalid[s] && ar_pick == LNC'(s);
  end
  always_ff @(posedge clk)
    if (rst) begin m_arvalid <= 1'b0; ar_last <= '0; end
    else if (!m_arvalid || m_arready) begin
      m_arvalid <= |s_arvalid;
      if (|s_arvalid) ar_last <= ar_pick;
    end
  always_ff @(posedge clk)                 // (the payload: no reset)
    if ((!m_arvalid || m_arready) && |s_arvalid) begin
      m_arid   <= {ar_pick, (IDW - LNC)'(0)};
      m_araddr <= bhw(dimm(s_araddr[ar_pick], ar_pick));
      m_arlen  <= 8'(s_arlen[ar_pick] >> 1);
    end

  // ---- AW: the same, each grant queued for W (u_wo); none while u_wo is full
  logic [LNC-1:0] aw_last, aw_pick, wo_m;
  logic aw_room, aw_go, wo_rv, wo_wr;
  assign aw_pick = rr(s_awvalid, aw_last);
  assign aw_room = (!m_awvalid || m_awready) && wo_wr;
  assign aw_go   = aw_room && |s_awvalid;
  for (genvar s = 0; s < NC; s++) begin : g_awr
    assign s_awready[s] = aw_room && s_awvalid[s] && aw_pick == LNC'(s);
  end
  always_ff @(posedge clk)
    if (rst) begin m_awvalid <= 1'b0; aw_last <= '0; end
    else if (!m_awvalid || m_awready) begin
      m_awvalid <= aw_go;
      if (aw_go) aw_last <= aw_pick;
    end
  always_ff @(posedge clk)
    if ((!m_awvalid || m_awready) && aw_go) begin
      m_awid   <= {aw_pick, (IDW - LNC)'(0)};
      m_awaddr <= bhw(dimm(s_awaddr[aw_pick], aw_pick));
      m_awlen  <= 8'(s_awlen[aw_pick] >> 1);
    end
  logic [WOC-1:0][LNC-1:0] wo_c;           // the head's copies (WOC)
  otpu_sfifo #(.W(LNC * WOC), .DEPTH(16), .RO(1'b1), .HCP(WOC)) u_wo (.clk, .rst, .wvalid(aw_go),
    .wready(wo_wr), .wdata({WOC{aw_pick}}), .rvalid(wo_rv),
    .rready(m_wvalid && m_wready && m_wlast), .rdata(wo_c));
  assign wo_m = wo_c[0];

  // ---- W: two 256-bit beats of the head's bridge into one 512-bit beat
  logic         hv;                        // the first half held
  logic [255:0] hd;
  logic [31:0]  hs;
  logic         wv;
  logic [NC-1:0][287:0] w_s;               // each bridge's strobes and data
  logic [287:0] w_h;                       // the head's bridge's, by wo_c's copies
  localparam int WG = 288 / WOC;
  for (genvar s = 0; s < NC; s++) begin : g_ws
    assign w_s[s] = {s_wstrb[s], s_wdata[s]};
  end
  for (genvar g = 0; g < WOC; g++) begin : g_wh
    logic [NC-1:0][WG-1:0] c;
    for (genvar s = 0; s < NC; s++) begin : g_s
      assign c[s] = w_s[s][g * WG +: WG];
    end
    assign w_h[g * WG +: WG] = c[wo_c[g]];
  end
  assign wv = wo_rv && s_wvalid[wo_m];
  assign m_wvalid = wv && hv;
  assign m_wdata  = {w_h[255:0], hd};
  assign m_wstrb  = {w_h[287:256], hs};
  assign m_wlast  = s_wlast[wo_m];
  for (genvar s = 0; s < NC; s++) begin : g_wr
    assign s_wready[s] = wo_rv && wo_m == LNC'(s) && (!hv || m_wready);
  end
  always_ff @(posedge clk)
    if (rst) hv <= 1'b0;
    else if (wv && !hv) hv <= 1'b1;
    else if (m_wvalid && m_wready) hv <= 1'b0;
  always_ff @(posedge clk)
    if (wv && !hv) begin hd <= w_h[255:0]; hs <= w_h[287:256]; end

  // ---- B: back by BID's top bits
  logic [LNC-1:0] b_s;
  assign b_s = m_bid[IDW-1 -: LNC];
  for (genvar s = 0; s < NC; s++) begin : g_b
    assign s_bvalid[s] = m_bvalid && b_s == LNC'(s);
  end
  assign s_bresp  = m_bresp;
  assign m_bready = s_bready[b_s];

  // ---- R: by RID's top bits into the bridge's FIFO; out as two halves a beat. The FIFO is RL lanes
  // (otpu_mig_pair_rl, each a hierarchy of its own: its FIFO's pointers and flags and its half
  // select its own copies, which synthesis would otherwise merge), lane l bits [LW l +: LW] of both
  // 256-bit halves, lane 0 RLAST and RRESP too: one pop and one read address drove all 515 bits' head
  // registers and LUT RAMs, and at 300 MHz their routes alone took 3.6-3.8 ns (dev100, 2026-10-28:
  // aclk's worst paths). The lanes take the same pushes and pops, so lane 0's flags are the FIFO's.
  localparam int RL = 4, LW = 256 / RL;
  logic [LNC-1:0] r_s;
  logic [NC-1:0]  rf_wr;                   // RS 0: the FIFO not full; 1: its credits not all out
  assign r_s = m_rid[IDW-1 -: LNC];
  assign m_rready = rf_wr[r_s];
  for (genvar s = 0; s < NC; s++) begin : g_r
    logic [RL-1:0] rv, wr, hi;
    logic [2:0]    rx;                     // lane 0's RLAST and RRESP
    logic          pv;                     // the FIFO's push
    logic [514:0]  pd;                     // and its beat, RLAST and RRESP
    if (RS) begin : g_rs
      (* keep = "true" *) logic [514:0] sd;
      logic sv = 1'b0, nf = 1'b0;          // the stage holds a beat; all RD credits out
      localparam int NW = $clog2(RD + 1);
      logic [NW-1:0] n = '0;               // beats accepted and not yet popped
      logic acc, pop;
      assign acc = m_rvalid && m_rready && r_s == LNC'(s);
      assign pop = rv[0] && hi[0] && s_rready[s];
      always_ff @(posedge clk) sd <= {m_rlast, m_rresp, m_rdata};
      always_ff @(posedge clk)
        if (rst) begin sv <= 1'b0; n <= '0; nf <= 1'b0; end
        else begin
          sv <= acc;
          n  <= n + NW'(acc) - NW'(pop);
          nf <= int'(n) + int'(acc) - int'(pop) == RD;
        end
      assign pv = sv;
      assign pd = sd;
      assign rf_wr[s] = !nf;
`ifndef SYNTHESIS
      always_ff @(posedge clk)
        if (!rst && sv && !wr[0])
          $error("otpu_mig_pair: bridge %0d's R stage into a full FIFO (credits %0d)", s, n);
`endif
    end else begin : g_rd
      assign pv = m_rvalid && r_s == LNC'(s);
      assign pd = {m_rlast, m_rresp, m_rdata};
      assign rf_wr[s] = wr[0];
    end
    for (genvar l = 0; l < RL; l++) begin : g_l
      localparam int XW = l == 0 ? 3 : 0;
      logic [XW+2*LW-1:0] w;
      logic [XW+LW-1:0]   q;
      if (l == 0) begin : g_x
        assign w  = {pd[514:512], pd[256 +: LW], pd[0 +: LW]};
        assign rx = q[LW +: 3];
      end else begin : g_d
        assign w  = {pd[256 + LW * l +: LW], pd[LW * l +: LW]};
      end
      (* keep_hierarchy = "yes" *)
      otpu_mig_pair_rl #(.LW(LW), .XW(XW), .RD(RD)) u_rl (.clk, .rst,
        .wvalid(pv), .wready(wr[l]), .wdata(w),
        .rvalid(rv[l]), .rready(s_rready[s]), .rdata(q), .hi(hi[l]));
      assign s_rdata[s][LW * l +: LW] = q[LW-1:0];
    end
    assign s_rvalid[s] = rv[0];
    assign s_rresp[s]  = rx[1:0];
    assign s_rlast[s]  = rx[2] && hi[0];
`ifndef SYNTHESIS
    always_ff @(posedge clk)
      if (!rst && (rv != {RL{rv[0]}} || wr != {RL{wr[0]}} || hi != {RL{hi[0]}}))
        $error("otpu_mig_pair: bridge %0d's R lanes out of step (rvalid %b, wready %b, high %b)",
               s, rv, wr, hi);
`endif
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst) begin
      for (int s = 0; s < NC; s++) begin
        if (s_arvalid[s] && (!s_arlen[s][0] || s_araddr[s][5:0] != 6'd0 || s_araddr[s][33:CL] != '0))
          $error("otpu_mig_pair: bridge %0d's AR (addr %h, len %0d): not 512-bit beats in 2^%0d bytes",
                 s, s_araddr[s], s_arlen[s], CL);
        if (s_awvalid[s] && (!s_awlen[s][0] || s_awaddr[s][5:0] != 6'd0 || s_awaddr[s][33:CL] != '0))
          $error("otpu_mig_pair: bridge %0d's AW (addr %h, len %0d): not 512-bit beats in 2^%0d bytes",
                 s, s_awaddr[s], s_awlen[s], CL);
        if (s_rvalid[s] && !s_rready[s])
          $error("otpu_mig_pair: bridge %0d held R back (otpu_hbm_ch takes every beat)", s);
      end
      if (wv && !hv && s_wlast[wo_m])
        $error("otpu_mig_pair: WLAST on a 512-bit beat's first half");
    end
`endif
endmodule

// One lane of otpu_mig_pair's R FIFO (its own pointers, flags and half select): entries of XW extra
// bits (lane 0's RLAST and RRESP), LW bits of a 512-bit beat's high 256-bit half and LW of its low
// half; out (rdata) the extra bits and the low half's LW bits, then (hi) the high half's. A beat
// leaves once its high half is taken.
module otpu_mig_pair_rl #(
  parameter int LW = 64,
  parameter int XW = 0,
  parameter int RD = 32
) (
  input  logic               clk,
  input  logic               rst,
  input  logic               wvalid,
  output logic               wready,
  input  logic [XW+2*LW-1:0] wdata,
  output logic               rvalid,
  input  logic               rready,
  output logic [XW+LW-1:0]   rdata,
  output logic               hi            // rdata is the high half's
);
  logic [XW+2*LW-1:0] d;
  otpu_sfifo #(.W(XW + 2 * LW), .DEPTH(RD), .RO(1'b1)) u_rf (.clk, .rst, .wvalid, .wready,
    .wdata, .rvalid, .rready(rvalid && hi && rready), .rdata(d));
  assign rdata[LW-1:0] = hi ? d[2*LW-1:LW] : d[LW-1:0];
  if (XW > 0) begin : g_x
    assign rdata[LW +: XW] = d[2*LW +: XW];
  end
  always_ff @(posedge clk)
    if (rst) hi <= 1'b0;
    else if (rvalid && rready) hi <= !hi;
endmodule
