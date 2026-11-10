// AXI plumbing of the builds whose channels are otpu_hbm_ch bridges (the BCU1525's WG builds,
// rtl/boards/bcu1525/otpu_bcu_wg_sys.sv): a register slice for one valid / ready channel
// (otpu_rs), an AXI3 port's register slices (otpu_axi3_rs: N of them in a row, for the routes
// from the bridges to the memory's ports), and the mux of one port between two masters
// (otpu_axi3_mux2). A far slice's channels cross the SLRs inside their bridges (otpu_hbm_ch's
// pipeline between its halves, otpu_hbm_ln).

// One channel's register slice, a beat a cycle, two entries; valid and ready leave from
// flip-flops. RING 1: the entries in a ring, the data from the read entry (a 2:1 mux on
// flip-flops); while not full the entry the next push takes loads s_data every cycle, so the
// entries' enables are the ring's state alone: s_valid and m_ready reach three flip-flops (wp or
// rp, ne, fu), never the data (the HBM ports' slices: as a skid buffer, probe build 3 placed, HBM
// port 0's WREADY into ~580 data enables was -2.35 ns at 450 MHz). RING 0: a skid buffer, the data
// from an output register (flip-flop to flip-flop across an SLR; its
// enables take m_ready). The handshakes are the same, cycle for cycle. No reset on the data (the
// HBM AXI reset into every port's data enables: probe build 2, -2.2 ns). RCP > 0 (RING): the read
// entry's select in RCP copies of rp (DONT_TOUCH), each over W / RCP of the data's bits (the
// BCU1525's two-slice host mux, otpu_mig_hmux: the 519-bit R ring's rp drove 521 loads through
// SLR0's DDR4 controller, -0.229 ns at 300 MHz in s2syn); 0, the default: rp alone. WCP > 0
// (RING): the entries' enables (!fu && !wp, !fu && wp) as WCP pairs of registers (DONT_TOUCH),
// each over W / WCP of the data's bits, loaded with the enables' next values (wp's and fu's, the
// reset's included), so each is the enable cycle for cycle (B1, bcu1525-x2: a bridge's 289-bit W
// ring's fu into its entries' 289 enables, and the host mux's 577-bit W ring's into 577); 0, the
// default: fu and wp's.
module otpu_rs #(
  parameter int W = 8,
  parameter bit RING = 1'b1,
  parameter int RCP = 0,
  parameter int WCP = 0
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         s_valid,
  output logic         s_ready,
  input  logic [W-1:0] s_data,
  output logic         m_valid,
  input  logic         m_ready,
  output logic [W-1:0] m_data
);
  if (RING) begin : g_ring
    logic         wp = 1'b0, rp = 1'b0;    // the entry the next push / pop takes
    logic         ne = 1'b0, fu = 1'b0;    // one entry or two, two
    logic [W-1:0] e0, e1;
    logic         push, pop;
    assign push    = s_valid && !fu;
    assign pop     = ne && m_ready;
    assign s_ready = !fu;
    assign m_valid = ne;
    logic [W-1:0] m_dc;                    // RCP: the data by the copies
    assign m_data  = RCP > 0 ? m_dc : rp ? e1 : e0;
    if (RCP > 0) begin : g_rcp
      localparam int GW = (W + RCP - 1) / RCP;
      initial if ((RCP - 1) * GW >= W) $fatal(1, "otpu_rs: RCP %0d over %0d bits leaves a copy none", RCP, W);
      (* dont_touch = "true" *) logic [RCP-1:0] rpc = '0;
      always_ff @(posedge clk) rpc <= rst ? '0 : rpc ^ {RCP{pop}};
      for (genvar g = 0; g < RCP; g++) begin : g_c
        localparam int LO = g * GW, HI = (g + 1) * GW < W ? (g + 1) * GW : W;
        assign m_dc[HI-1:LO] = rpc[g] ? e1[HI-1:LO] : e0[HI-1:LO];
      end
    end else begin : g_rp
      assign m_dc = '0;
    end
    always_ff @(posedge clk)
      if (rst) begin wp <= 1'b0; rp <= 1'b0; ne <= 1'b0; fu <= 1'b0; end
      else begin
        wp <= wp ^ push;
        rp <= rp ^ pop;
        ne <= fu || push || (ne && !pop);
        fu <= fu ? !pop : ne && push && !pop;
      end
    if (WCP > 0) begin : g_wcp
      localparam int GW = (W + WCP - 1) / WCP;
      initial if ((WCP - 1) * GW >= W) $fatal(1, "otpu_rs: WCP %0d over %0d bits leaves a copy none", WCP, W);
      logic wpn, fun;                      // wp's and fu's next values
      assign wpn = !rst && (wp ^ push);
      assign fun = !rst && (fu ? !pop : ne && push && !pop);
      (* dont_touch = "true" *) logic [WCP-1:0] we0 = '1, we1 = '0;
      always_ff @(posedge clk) begin
        we0 <= {WCP{!fun && !wpn}};
        we1 <= {WCP{!fun && wpn}};
      end
      always_ff @(posedge clk)
        for (int b = 0; b < W; b++) begin
          if (we0[b / GW]) e0[b] <= s_data[b];
          if (we1[b / GW]) e1[b] <= s_data[b];
        end
`ifndef SYNTHESIS
      always_ff @(posedge clk)
        if (we0 != {WCP{!fu && !wp}} || we1 != {WCP{!fu && wp}})
          $fatal(1, "otpu_rs: the enables' copies are not the ring's");
`endif
    end else begin : g_we
      always_ff @(posedge clk) begin
        if (!fu && !wp) e0 <= s_data;
        if (!fu && wp)  e1 <= s_data;
      end
    end
  end else begin : g_skid
    logic         v0 = 1'b0, v1 = 1'b0;    // the output register, the skid entry
    logic [W-1:0] d0, d1;
    assign m_valid = v0;
    assign m_data  = d0;
    assign s_ready = !v1;
    always_ff @(posedge clk)
      if (rst) begin v0 <= 1'b0; v1 <= 1'b0; end
      else if (!v0 || m_ready) begin
        v0 <= v1 || s_valid;
        v1 <= 1'b0;
      end else if (s_valid && !v1) v1 <= 1'b1;
    always_ff @(posedge clk)
      if (!v0 || m_ready) d0 <= v1 ? d1 : s_data;
      else if (s_valid && !v1) d1 <= s_data;
  end
endmodule

// N register slices on each channel of an HBM AXI3 port (256-bit, 6-bit IDs; N = 0: wires).
// RBP: R and B as plain flip-flop stages (no ready: RREADY / BREADY 1 at the port), for a slave
// side that takes every beat (a bridge's, rready / bready 1; a simulation $error if it holds one
// back): the port's RDATA a flip-flop each, instead of a ring's two entries (probe build 7
// routed: the BLIs' R into the rings' entries -0.15 ns at 450 MHz).
module otpu_axi3_rs #(
  parameter int N = 1,
  parameter bit RBP = 1'b0,
  parameter int WCP = 0,                   // the W rings' entry enables in copies (otpu_rs WCP)
  parameter int RCP = 0                    // and their read selects (otpu_rs RCP)
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         s_awvalid,
  output logic         s_awready,
  input  logic [5:0]   s_awid,
  input  logic [33:0]  s_awaddr,
  input  logic [3:0]   s_awlen,
  input  logic         s_wvalid,
  output logic         s_wready,
  input  logic [255:0] s_wdata,
  input  logic [31:0]  s_wstrb,
  input  logic         s_wlast,
  output logic         s_bvalid,
  input  logic         s_bready,
  output logic [5:0]   s_bid,
  output logic [1:0]   s_bresp,
  input  logic         s_arvalid,
  output logic         s_arready,
  input  logic [5:0]   s_arid,
  input  logic [33:0]  s_araddr,
  input  logic [3:0]   s_arlen,
  output logic         s_rvalid,
  input  logic         s_rready,
  output logic [5:0]   s_rid,
  output logic [255:0] s_rdata,
  output logic [1:0]   s_rresp,
  output logic         s_rlast,
  output logic         m_awvalid,
  input  logic         m_awready,
  output logic [5:0]   m_awid,
  output logic [33:0]  m_awaddr,
  output logic [3:0]   m_awlen,
  output logic         m_wvalid,
  input  logic         m_wready,
  output logic [255:0] m_wdata,
  output logic [31:0]  m_wstrb,
  output logic         m_wlast,
  input  logic         m_bvalid,
  output logic         m_bready,
  input  logic [5:0]   m_bid,
  input  logic [1:0]   m_bresp,
  output logic         m_arvalid,
  input  logic         m_arready,
  output logic [5:0]   m_arid,
  output logic [33:0]  m_araddr,
  output logic [3:0]   m_arlen,
  input  logic         m_rvalid,
  output logic         m_rready,
  input  logic [5:0]   m_rid,
  input  logic [255:0] m_rdata,
  input  logic [1:0]   m_rresp,
  input  logic         m_rlast
);
  localparam int AW = 44, WW = 289, BW = 8, RW = 265;
  logic [N:0] awv, awr, wv, wr, bv, br, arv, arr, rv, rr;
  logic [N:0][AW-1:0] awd, ard;
  logic [N:0][WW-1:0] wd;
  logic [N:0][BW-1:0] bd;
  logic [N:0][RW-1:0] rd;
  // stage 0 is the slave side, stage N the master side
  assign awv[0] = s_awvalid; assign s_awready = awr[0]; assign awd[0] = {s_awid, s_awaddr, s_awlen};
  assign wv[0]  = s_wvalid;  assign s_wready  = wr[0];  assign wd[0]  = {s_wdata, s_wstrb, s_wlast};
  assign arv[0] = s_arvalid; assign s_arready = arr[0]; assign ard[0] = {s_arid, s_araddr, s_arlen};
  assign s_bvalid = bv[0]; assign br[0] = s_bready; assign {s_bid, s_bresp} = bd[0];
  assign s_rvalid = rv[0]; assign rr[0] = s_rready; assign {s_rid, s_rdata, s_rresp, s_rlast} = rd[0];
  assign m_awvalid = awv[N]; assign awr[N] = m_awready; assign {m_awid, m_awaddr, m_awlen} = awd[N];
  assign m_wvalid  = wv[N];  assign wr[N]  = m_wready;  assign {m_wdata, m_wstrb, m_wlast} = wd[N];
  assign m_arvalid = arv[N]; assign arr[N] = m_arready; assign {m_arid, m_araddr, m_arlen} = ard[N];
  assign bv[N] = m_bvalid; assign m_bready = br[N]; assign bd[N] = {m_bid, m_bresp};
  assign rv[N] = m_rvalid; assign m_rready = rr[N]; assign rd[N] = {m_rid, m_rdata, m_rresp, m_rlast};
  for (genvar i = 0; i < N; i++) begin : g_st
    otpu_rs #(.W(AW)) u_aw (.clk, .rst, .s_valid(awv[i]), .s_ready(awr[i]), .s_data(awd[i]),
      .m_valid(awv[i + 1]), .m_ready(awr[i + 1]), .m_data(awd[i + 1]));
    otpu_rs #(.W(WW), .RCP(RCP), .WCP(WCP)) u_w (.clk, .rst, .s_valid(wv[i]), .s_ready(wr[i]), .s_data(wd[i]),
      .m_valid(wv[i + 1]), .m_ready(wr[i + 1]), .m_data(wd[i + 1]));
    otpu_rs #(.W(AW)) u_ar (.clk, .rst, .s_valid(arv[i]), .s_ready(arr[i]), .s_data(ard[i]),
      .m_valid(arv[i + 1]), .m_ready(arr[i + 1]), .m_data(ard[i + 1]));
    if (RBP) begin : g_rbp
      logic          bq = 1'b0, rq = 1'b0;
      logic [BW-1:0] bdq;
      logic [RW-1:0] rdq;
      always_ff @(posedge clk) begin
        bq <= bv[i + 1] && !rst; bdq <= bd[i + 1];
        rq <= rv[i + 1] && !rst; rdq <= rd[i + 1];
      end
      assign bv[i] = bq; assign bd[i] = bdq; assign br[i + 1] = 1'b1;
      assign rv[i] = rq; assign rd[i] = rdq; assign rr[i + 1] = 1'b1;
`ifndef SYNTHESIS
      always_ff @(posedge clk)
        if (!rst && ((bq && !br[i]) || (rq && !rr[i])))
          $error("otpu_axi3_rs: RBP, and the slave side held R or B back");
`endif
    end else begin : g_rb
      otpu_rs #(.W(BW)) u_b (.clk, .rst, .s_valid(bv[i + 1]), .s_ready(br[i + 1]), .s_data(bd[i + 1]),
        .m_valid(bv[i]), .m_ready(br[i]), .m_data(bd[i]));
      otpu_rs #(.W(RW)) u_r (.clk, .rst, .s_valid(rv[i + 1]), .s_ready(rr[i + 1]), .s_data(rd[i + 1]),
        .m_valid(rv[i]), .m_ready(rr[i]), .m_data(rd[i]));
    end
  end
endmodule

// One AXI3 port shared by two masters without IDs: s0 (a channel's bridge, otpu_hbm_ch) gets
// ID 0x00, s1 (the host's path) ID 0x20. AR and AW alternate between the masters that ask
// (one burst a turn, from a register); B and R go back by their ID's bit 5 (the HBM may reorder
// between the two IDs, never within one). W follows AW's order (the HBM's port has no WID): each
// granted AW queues its master in wo, and W passes from wo's head master until its WLAST. A master
// holds its AW until its W follows (otpu_hbm_ch issues an AW only with all its data in; the
// SmartConnect's W follows its AW), so W never waits on the other master's data for long.
module otpu_axi3_mux2 (
  input  logic              clk,
  input  logic              rst,
  input  logic [1:0]        s_awvalid,
  output logic [1:0]        s_awready,
  input  logic [1:0][33:0]  s_awaddr,
  input  logic [1:0][3:0]   s_awlen,
  input  logic [1:0]        s_wvalid,
  output logic [1:0]        s_wready,
  input  logic [1:0][255:0] s_wdata,
  input  logic [1:0][31:0]  s_wstrb,
  input  logic [1:0]        s_wlast,
  output logic [1:0]        s_bvalid,
  input  logic [1:0]        s_bready,
  output logic [1:0]        s_bresp,     // both masters' (one valid at a time)
  input  logic [1:0]        s_arvalid,
  output logic [1:0]        s_arready,
  input  logic [1:0][33:0]  s_araddr,
  input  logic [1:0][3:0]   s_arlen,
  output logic [1:0]        s_rvalid,
  input  logic [1:0]        s_rready,
  output logic [255:0]      s_rdata,
  output logic [1:0]        s_rresp,
  output logic              s_rlast,
  output logic              m_awvalid,
  input  logic              m_awready,
  output logic [5:0]        m_awid,
  output logic [33:0]       m_awaddr,
  output logic [3:0]        m_awlen,
  output logic              m_wvalid,
  input  logic              m_wready,
  output logic [255:0]      m_wdata,
  output logic [31:0]       m_wstrb,
  output logic              m_wlast,
  input  logic              m_bvalid,
  output logic              m_bready,
  input  logic [5:0]        m_bid,
  input  logic [1:0]        m_bresp,
  output logic              m_arvalid,
  input  logic              m_arready,
  output logic [5:0]        m_arid,
  output logic [33:0]       m_araddr,
  output logic [3:0]        m_arlen,
  input  logic              m_rvalid,
  output logic              m_rready,
  input  logic [5:0]        m_rid,
  input  logic [255:0]      m_rdata,
  input  logic [1:0]        m_rresp,
  input  logic              m_rlast
);
  // ---- AR: a register, loaded from the master whose turn it is (or the other one, if only it asks)
  logic ar_last, ar_pick;
  assign ar_pick = s_arvalid[1] && (!s_arvalid[0] || !ar_last);
  assign s_arready[0] = (!m_arvalid || m_arready) && s_arvalid[0] && !ar_pick;
  assign s_arready[1] = (!m_arvalid || m_arready) && ar_pick;
  always_ff @(posedge clk)
    if (rst) begin m_arvalid <= 1'b0; ar_last <= 1'b0; end
    else if (!m_arvalid || m_arready) begin
      m_arvalid <= |s_arvalid;
      if (|s_arvalid) ar_last <= ar_pick;
    end
  always_ff @(posedge clk)                 // (the payload: no reset, as otpu_rs's)
    if ((!m_arvalid || m_arready) && |s_arvalid) begin
      m_arid <= {ar_pick, 5'd0}; m_araddr <= s_araddr[ar_pick]; m_arlen <= s_arlen[ar_pick];
    end

  // ---- AW: the same, each grant queued for W (wo); none while wo is full
  logic aw_last, aw_pick, aw_room, aw_go, wo_rv, wo_wr;
  logic wo_m;                              // the master whose W goes now
  assign aw_pick = s_awvalid[1] && (!s_awvalid[0] || !aw_last);
  assign aw_room = (!m_awvalid || m_awready) && wo_wr;
  assign aw_go   = aw_room && |s_awvalid;
  assign s_awready[0] = aw_room && s_awvalid[0] && !aw_pick;
  assign s_awready[1] = aw_room && aw_pick;
  always_ff @(posedge clk)
    if (rst) begin m_awvalid <= 1'b0; aw_last <= 1'b0; end
    else if (!m_awvalid || m_awready) begin
      m_awvalid <= aw_go;
      if (aw_go) aw_last <= aw_pick;
    end
  always_ff @(posedge clk)
    if ((!m_awvalid || m_awready) && aw_go) begin
      m_awid <= {aw_pick, 5'd0}; m_awaddr <= s_awaddr[aw_pick]; m_awlen <= s_awlen[aw_pick];
    end
  otpu_sfifo #(.W(1), .DEPTH(16), .RO(1'b1)) u_wo (.clk, .rst, .wvalid(aw_go), .wready(wo_wr),
    .wdata(aw_pick), .rvalid(wo_rv), .rready(m_wvalid && m_wready && m_wlast), .rdata(wo_m));
  assign m_wvalid    = wo_rv && s_wvalid[wo_m];
  assign m_wdata     = s_wdata[wo_m];
  assign m_wstrb     = s_wstrb[wo_m];
  assign m_wlast     = s_wlast[wo_m];
  assign s_wready[0] = wo_rv && !wo_m && m_wready;
  assign s_wready[1] = wo_rv && wo_m && m_wready;

  // ---- B and R: back by the ID's bit 5
  assign s_bvalid = {m_bvalid && m_bid[5], m_bvalid && !m_bid[5]};
  assign s_bresp  = m_bresp;
  assign m_bready = s_bready[m_bid[5]];
  assign s_rvalid = {m_rvalid && m_rid[5], m_rvalid && !m_rid[5]};
  assign s_rdata  = m_rdata;
  assign s_rresp  = m_rresp;
  assign s_rlast  = m_rlast;
  assign m_rready = s_rready[m_rid[5]];
endmodule
