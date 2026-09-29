// One DDR3 channel's user side: the accelerator's AXI master (otpu_axi_dram, core clock,
// 512-bit) and XDMA's (host loads and reads, axi_aclk, 128-bit) onto the MIG's native interface
// (ui_clk). With otpu_axi_split2 it replaces the SmartConnect and the MIG's AXI front end.
//
// Every 64-byte beat becomes one native command (BL8 at 4:1: one app_en, and for a write one
// app_wdf_wren with app_wdf_end), so a burst costs nothing per transaction. Each master's
// commands cross into ui_clk through its own asynchronous FIFOs (commands, write data) and its
// read data comes back through another; an arbiter in ui_clk takes up to RUN commands from one
// master before it lets the other in. The MIG returns read data in command order (its read
// buffer reorders), so one tag per read (its master) routes the data back. A master's read is
// issued only when its read-data FIFO has room for it and for every read still in flight
// (credits), because app_rd_data cannot be held back.
//
// Writes: a set app_wdf_mask bit keeps the byte. A beat with any byte kept goes out as the ECC
// controller's wr_bytes command (a read-modify-write in the MIG, as its AXI front end does for a
// partial strobe); a whole beat as a plain write. XDMA's 16-byte beats are packed into 64-byte
// beats per burst (a partial beat at either end keeps the missing lanes). A write burst gets its
// response once all of its beats are taken by the MIG (app_rdy): the MIG keeps the order of
// accesses to one address, so a read issued after a response sees the write, and the host's
// reads after the accelerator halts (every write answered: otpu_axi_dram's wr_idle) see its data.
//
// Addresses: bits 30:6 of the byte address select the 64-byte beat in the channel (2 GiB; bit
// 31, the channel, is dropped). app_addr = {rank 0, beat, 3'b000}. The accelerator's bursts are
// INCR of 64-byte beats; XDMA's are INCR of 16-byte beats (AxSIZE 4) and may start anywhere in a
// 64-byte beat. Narrow bursts are not supported (neither master issues them). Reads return in
// order (legal for any IDs); responses are always OKAY.
//
// Resets: each master's path has a hold in ui_clk -- the MIG's reset, or the master's own reset
// (synchronized), then 16 cycles, and on until its reads in flight have come back (their data is
// dropped) -- and the same hold, synchronized, in the master's clock. The clock-crossing FIFOs
// reset only on the hold, in both clocks, so neither side ever sees the other's pointers reset
// under it; the master's side of the bridge resets on its reset or the hold. So a PCIe link reset
// (XDMA's axi_aresetn) while the accelerator runs costs neither the other master nor the memory.
module otpu_mig_ch #(
  parameter int XIDW = 4,         // XDMA M_AXI ID width
  parameter int ARD  = 64,        // accelerator read-data FIFO (read credits), beats
  parameter int XRD  = 16,        // XDMA read-data FIFO, beats
  parameter int RUN  = 32         // commands from one master before the arbiter may switch
) (
  // accelerator, core clock
  input  logic            clk,
  input  logic            rst,
  input  logic            s_awvalid,
  output logic            s_awready,
  input  logic [0:0]      s_awid,
  input  logic [31:0]     s_awaddr,
  input  logic [7:0]      s_awlen,
  input  logic            s_wvalid,
  output logic            s_wready,
  input  logic [511:0]    s_wdata,
  input  logic [63:0]     s_wstrb,
  input  logic            s_wlast,
  output logic            s_bvalid,
  input  logic            s_bready,
  output logic [0:0]      s_bid,
  output logic [1:0]      s_bresp,
  input  logic            s_arvalid,
  output logic            s_arready,
  input  logic [0:0]      s_arid,
  input  logic [31:0]     s_araddr,
  input  logic [7:0]      s_arlen,
  output logic            s_rvalid,
  input  logic            s_rready,
  output logic [0:0]      s_rid,
  output logic [511:0]    s_rdata,
  output logic [1:0]      s_rresp,
  output logic            s_rlast,
  // XDMA, axi_aclk
  input  logic            xclk,
  input  logic            xrst,
  input  logic            x_awvalid,
  output logic            x_awready,
  input  logic [XIDW-1:0] x_awid,
  input  logic [31:0]     x_awaddr,
  input  logic [7:0]      x_awlen,
  input  logic            x_wvalid,
  output logic            x_wready,
  input  logic [127:0]    x_wdata,
  input  logic [15:0]     x_wstrb,
  input  logic            x_wlast,
  output logic            x_bvalid,
  input  logic            x_bready,
  output logic [XIDW-1:0] x_bid,
  output logic [1:0]      x_bresp,
  input  logic            x_arvalid,
  output logic            x_arready,
  input  logic [XIDW-1:0] x_arid,
  input  logic [31:0]     x_araddr,
  input  logic [7:0]      x_arlen,
  output logic            x_rvalid,
  input  logic            x_rready,
  output logic [XIDW-1:0] x_rid,
  output logic [127:0]    x_rdata,
  output logic [1:0]      x_rresp,
  output logic            x_rlast,
  // MIG native interface, ui_clk
  input  logic            uclk,
  input  logic            urst,
  output logic [28:0]     app_addr,
  output logic [2:0]      app_cmd,
  output logic            app_en,
  input  logic            app_rdy,
  output logic [511:0]    app_wdf_data,
  output logic [63:0]     app_wdf_mask,
  output logic            app_wdf_wren,
  output logic            app_wdf_end,
  input  logic            app_wdf_rdy,
  input  logic [511:0]    app_rd_data,
  input  logic            app_rd_data_valid
);
  localparam int QW = 27;                  // command: {write, partial, beat[24:0]}
  localparam int DW = 576;                 // write data: {mask[63:0], data[511:0]}
  localparam int CW = 16;                  // write-accept counters (beats, modulo 2^16)
  localparam int AOW = $clog2(ARD) + 1;
  localparam int XOW = $clog2(XRD) + 1;

  function automatic logic [CW-1:0] b2g(input logic [CW-1:0] b);
    return b ^ (b >> 1);
  endfunction
  function automatic logic [CW-1:0] g2b(input logic [CW-1:0] g);
    logic [CW-1:0] b;
    b[CW-1] = g[CW-1];
    for (int i = CW - 2; i >= 0; i--) b[i] = b[i + 1] ^ g[i];
    return b;
  endfunction

  // ================================================================ resets
  (* ASYNC_REG = "TRUE" *) logic a_rs1, a_rs2, x_rs1, x_rs2;    // master resets in uclk
  (* ASYNC_REG = "TRUE" *) logic a_hs1 = 1'b1, a_hs2 = 1'b1;     // holds in the masters' clocks
  (* ASYNC_REG = "TRUE" *) logic x_hs1 = 1'b1, x_hs2 = 1'b1;
  logic a_hold = 1'b1, x_hold = 1'b1;
  logic [3:0] a_hcnt, x_hcnt;
  logic a_crst = 1'b1, x_crst = 1'b1;      // the masters' sides of the bridge
  logic [AOW-1:0] a_out;                   // reads in flight (issued, data not yet in the FIFO)
  logic [XOW-1:0] x_out;
  always_ff @(posedge uclk) begin
    a_rs1 <= rst;  a_rs2 <= a_rs1;
    x_rs1 <= xrst; x_rs2 <= x_rs1;
    a_hcnt <= (urst || a_rs2) ? 4'hF : (a_hcnt != 0) ? a_hcnt - 1'b1 : a_hcnt;
    x_hcnt <= (urst || x_rs2) ? 4'hF : (x_hcnt != 0) ? x_hcnt - 1'b1 : x_hcnt;
    a_hold <= urst || a_rs2 || (a_hold && (a_hcnt != 0 || a_out != 0));
    x_hold <= urst || x_rs2 || (x_hold && (x_hcnt != 0 || x_out != 0));
  end
  always_ff @(posedge clk) begin
    a_hs1 <= a_hold; a_hs2 <= a_hs1;
    a_crst <= rst || a_hs2;
  end
  always_ff @(posedge xclk) begin
    x_hs1 <= x_hold; x_hs2 <= x_hs1;
    x_crst <= xrst || x_hs2;
  end

  // ================================================================ clock crossings
  logic          aq_wv, aq_wr, aq_rv, aq_rr;  logic [QW-1:0] aq_wd, aq_rd;
  logic          ad_wv, ad_wr, ad_rv, ad_rr;  logic [DW-1:0] ad_wd, ad_rd;
  logic          ar_wv, ar_wr, ar_rv, ar_rr;  logic [511:0]  ar_wd, ar_rd;
  logic          xq_wv, xq_wr, xq_rv, xq_rr;  logic [QW-1:0] xq_wd, xq_rd;
  logic          xd_wv, xd_wr, xd_rv, xd_rr;  logic [DW-1:0] xd_wd, xd_rd;
  logic          xr_wv, xr_wr, xr_rv, xr_rr;  logic [511:0]  xr_wd, xr_rd;
  logic [AOW-1:0] ar_used;
  logic [XOW-1:0] xr_used;
  logic [5:0] unused_aq, unused_ad;
  logic [4:0] unused_xq, unused_xd;

  otpu_afifo #(.W(QW), .DEPTH(32)) u_aq (.wclk(clk), .wrst(a_hs2), .wvalid(aq_wv), .wready(aq_wr),
    .wdata(aq_wd), .wused(unused_aq), .rclk(uclk), .rrst(a_hold), .rvalid(aq_rv), .rready(aq_rr),
    .rdata(aq_rd));
  otpu_afifo #(.W(DW), .DEPTH(32)) u_ad (.wclk(clk), .wrst(a_hs2), .wvalid(ad_wv), .wready(ad_wr),
    .wdata(ad_wd), .wused(unused_ad), .rclk(uclk), .rrst(a_hold), .rvalid(ad_rv), .rready(ad_rr),
    .rdata(ad_rd));
  otpu_afifo #(.W(512), .DEPTH(ARD)) u_ar (.wclk(uclk), .wrst(a_hold), .wvalid(ar_wv), .wready(ar_wr),
    .wdata(ar_wd), .wused(ar_used), .rclk(clk), .rrst(a_hs2), .rvalid(ar_rv), .rready(ar_rr),
    .rdata(ar_rd));
  otpu_afifo #(.W(QW), .DEPTH(16)) u_xq (.wclk(xclk), .wrst(x_hs2), .wvalid(xq_wv), .wready(xq_wr),
    .wdata(xq_wd), .wused(unused_xq), .rclk(uclk), .rrst(x_hold), .rvalid(xq_rv), .rready(xq_rr),
    .rdata(xq_rd));
  otpu_afifo #(.W(DW), .DEPTH(16)) u_xd (.wclk(xclk), .wrst(x_hs2), .wvalid(xd_wv), .wready(xd_wr),
    .wdata(xd_wd), .wused(unused_xd), .rclk(uclk), .rrst(x_hold), .rvalid(xd_rv), .rready(xd_rr),
    .rdata(xd_rd));
  otpu_afifo #(.W(512), .DEPTH(XRD)) u_xr (.wclk(uclk), .wrst(x_hold), .wvalid(xr_wv), .wready(xr_wr),
    .wdata(xr_wd), .wused(xr_used), .rclk(xclk), .rrst(x_hs2), .rvalid(xr_rv), .rready(xr_rr),
    .rdata(xr_rd));

  // write beats taken by the MIG, per master (gray counters from ui_clk)
  logic [CW-1:0] a_wacc, a_wacc_g, x_wacc, x_wacc_g;
  (* ASYNC_REG = "TRUE" *) logic [CW-1:0] a_wacc_s1, a_wacc_s2;
  (* ASYNC_REG = "TRUE" *) logic [CW-1:0] x_wacc_s1, x_wacc_s2;

  // ================================================================ accelerator side (clk)
  // AR and AW go into small issue queues (the beats to send) and descriptor queues (the R and B
  // channels' {id, len}); the issuers walk the head burst beat by beat.
  logic       ai_wv, ai_wr, ai_rv, ai_rr;  logic [32:0] ai_rd;     // {len, beat}
  logic       wi_wv, wi_wr, wi_rv, wi_rr;  logic [24:0] wi_rd;     // beat
  logic       rq_wv, rq_wr, rq_rv, rq_rr;  logic [8:0]  rq_rd;     // {id, len}
  logic       bq_wv, bq_wr, bq_rv, bq_rr;  logic [8:0]  bq_rd;
  otpu_sfifo #(.W(33), .DEPTH(4)) u_ai (.clk, .rst(a_crst), .wvalid(ai_wv), .wready(ai_wr),
    .wdata({s_arlen, s_araddr[30:6]}), .rvalid(ai_rv), .rready(ai_rr), .rdata(ai_rd));
  otpu_sfifo #(.W(25), .DEPTH(4)) u_wi (.clk, .rst(a_crst), .wvalid(wi_wv), .wready(wi_wr),
    .wdata(s_awaddr[30:6]), .rvalid(wi_rv), .rready(wi_rr), .rdata(wi_rd));
  otpu_sfifo #(.W(9), .DEPTH(64)) u_rq (.clk, .rst(a_crst), .wvalid(rq_wv), .wready(rq_wr),
    .wdata({s_arid, s_arlen}), .rvalid(rq_rv), .rready(rq_rr), .rdata(rq_rd));
  otpu_sfifo #(.W(9), .DEPTH(64)) u_bq (.clk, .rst(a_crst), .wvalid(bq_wv), .wready(bq_wr),
    .wdata({s_awid, s_awlen}), .rvalid(bq_rv), .rready(bq_rr), .rdata(bq_rd));
  assign s_arready = !a_crst && ai_wr && rq_wr;
  assign ai_wv     = s_arvalid && s_arready;
  assign rq_wv     = ai_wv;
  assign s_awready = !a_crst && wi_wr && bq_wr;
  assign wi_wv     = s_awvalid && s_awready;
  assign bq_wv     = wi_wv;

  // one command per cycle into the crossing: a read beat or a write beat, keeping to one
  // direction for up to 32 beats while the other waits (runs of reads, runs of writes)
  logic [7:0] a_roff, a_woff;
  logic       a_rcan, a_wcan, a_selw, a_lastw, a_rlast;
  logic [5:0] a_dirrun;
  assign a_rcan  = ai_rv;
  assign a_wcan  = wi_rv && s_wvalid && ad_wr;
  assign a_rlast = a_roff == ai_rd[32:25];
  always_comb begin
    if (a_rcan && a_wcan) a_selw = a_lastw ? (a_dirrun < 6'd32) : (a_dirrun >= 6'd32);
    else a_selw = a_wcan;
  end
  assign aq_wv    = !a_crst && aq_wr && (a_rcan || a_wcan);
  assign aq_wd    = a_selw ? {1'b1, ~&s_wstrb, 25'(wi_rd + a_woff)} : {2'b00, 25'(ai_rd[24:0] + a_roff)};
  assign s_wready = aq_wv && a_selw;
  assign ad_wv    = s_wvalid && s_wready;
  assign ad_wd    = {~s_wstrb, s_wdata};
  assign ai_rr    = aq_wv && !a_selw && a_rlast;
  assign wi_rr    = s_wvalid && s_wready && s_wlast;
  always_ff @(posedge clk) begin
    if (a_crst) begin
      a_roff <= '0; a_woff <= '0; a_lastw <= 1'b0; a_dirrun <= '0;
    end else if (aq_wv) begin
      if (a_selw) a_woff <= s_wlast ? '0 : a_woff + 1'b1;
      else        a_roff <= a_rlast ? '0 : a_roff + 1'b1;
      a_dirrun <= (a_selw != a_lastw) ? 6'd1 : (a_dirrun == 6'd63) ? a_dirrun : a_dirrun + 1'b1;
      a_lastw  <= a_selw;
    end
  end

  // R: the read data in order; id and last from the burst descriptors
  logic [7:0] a_rcnt;
  assign s_rvalid = ar_rv && rq_rv;
  assign s_rdata  = ar_rd;
  assign s_rid    = rq_rd[8];
  assign s_rlast  = a_rcnt == rq_rd[7:0];
  assign s_rresp  = 2'b00;
  assign ar_rr    = s_rvalid && s_rready;
  assign rq_rr    = s_rvalid && s_rready && s_rlast;
  always_ff @(posedge clk) begin
    if (a_crst) a_rcnt <= '0;
    else if (s_rvalid && s_rready) a_rcnt <= s_rlast ? '0 : a_rcnt + 1'b1;
  end

  // B: a burst's response once the MIG has taken all its beats
  logic [CW-1:0] a_wdone, a_wacc_c;
  always_ff @(posedge clk) begin
    if (a_crst) begin a_wacc_s1 <= '0; a_wacc_s2 <= '0; a_wacc_c <= '0; end
    else begin a_wacc_s1 <= a_wacc_g; a_wacc_s2 <= a_wacc_s1; a_wacc_c <= g2b(a_wacc_s2); end
  end
  assign s_bvalid = bq_rv && ((a_wacc_c - a_wdone) > CW'(bq_rd[7:0]));
  assign s_bid    = bq_rd[8];
  assign s_bresp  = 2'b00;
  assign bq_rr    = s_bvalid && s_bready;
  always_ff @(posedge clk) begin
    if (a_crst) a_wdone <= '0;
    else if (s_bvalid && s_bready) a_wdone <= a_wdone + CW'(bq_rd[7:0]) + 1'b1;
  end

  // ================================================================ XDMA side (xclk)
  // Read bursts: one command per 64-byte beat touched; the R channel hands out the 16-byte lanes
  // from the burst's first lane. Write bursts: 16-byte beats packed into 64-byte beats.
  localparam int XRQ = XIDW + 10;
  logic       xai_wv, xai_wr, xai_rv, xai_rr;  logic [31:0] xai_rd;      // {n - 1, beat}
  logic       xwi_wv, xwi_wr, xwi_rv, xwi_rr;  logic [26:0] xwi_rd;      // {lane, beat}
  logic       xrq_wv, xrq_wr, xrq_rv, xrq_rr;  logic [XRQ-1:0] xrq_rd;   // {id, len, lane}
  logic       xbq_wv, xbq_wr, xbq_rv, xbq_rr;  logic [XIDW+6:0] xbq_rd;  // {id, n - 1}
  logic [6:0] x_arn, x_awn;                // 64-byte beats of the burst, minus one
  assign x_arn = 7'(({1'b0, x_arlen} + {7'b0, x_araddr[5:4]}) >> 2);
  assign x_awn = 7'(({1'b0, x_awlen} + {7'b0, x_awaddr[5:4]}) >> 2);
  otpu_sfifo #(.W(32), .DEPTH(4)) u_xai (.clk(xclk), .rst(x_crst), .wvalid(xai_wv), .wready(xai_wr),
    .wdata({x_arn, x_araddr[30:6]}), .rvalid(xai_rv), .rready(xai_rr), .rdata(xai_rd));
  otpu_sfifo #(.W(27), .DEPTH(4)) u_xwi (.clk(xclk), .rst(x_crst), .wvalid(xwi_wv), .wready(xwi_wr),
    .wdata({x_awaddr[5:4], x_awaddr[30:6]}), .rvalid(xwi_rv), .rready(xwi_rr), .rdata(xwi_rd));
  otpu_sfifo #(.W(XRQ), .DEPTH(16)) u_xrq (.clk(xclk), .rst(x_crst), .wvalid(xrq_wv), .wready(xrq_wr),
    .wdata({x_arid, x_arlen, x_araddr[5:4]}), .rvalid(xrq_rv), .rready(xrq_rr), .rdata(xrq_rd));
  otpu_sfifo #(.W(XIDW + 7), .DEPTH(16)) u_xbq (.clk(xclk), .rst(x_crst), .wvalid(xbq_wv), .wready(xbq_wr),
    .wdata({x_awid, x_awn}), .rvalid(xbq_rv), .rready(xbq_rr), .rdata(xbq_rd));
  assign x_arready = !x_crst && xai_wr && xrq_wr;
  assign xai_wv    = x_arvalid && x_arready;
  assign xrq_wv    = xai_wv;
  assign x_awready = !x_crst && xwi_wr && xbq_wr;
  assign xwi_wv    = x_awvalid && x_awready;
  assign xbq_wv    = xwi_wv;

  // write packing: the burst's current 64-byte beat collects its lanes; it goes out with its
  // last lane or the burst's last beat. A completed write beat goes first, else a read beat.
  logic [6:0]   x_roff, x_woff;
  logic [1:0]   x_wlane_r, x_wlane;
  logic         x_wfirst, x_wpush, x_selw, x_rlastb;
  logic [511:0] x_wbuf, x_wdata_full;
  logic [63:0]  x_wstb, x_wstb_full;
  assign x_wlane  = x_wfirst ? xwi_rd[26:25] : x_wlane_r;
  assign x_wpush  = x_wlane == 2'd3 || x_wlast;
  assign x_selw   = xwi_rv && x_wvalid && x_wpush;
  assign x_rlastb = x_roff == xai_rd[31:25];
  always_comb begin
    x_wdata_full = x_wbuf;
    x_wstb_full  = x_wstb;
    x_wdata_full[x_wlane * 128 +: 128] = x_wdata;
    x_wstb_full[x_wlane * 16 +: 16]    = x_wstrb;
  end
  assign xq_wv    = !x_crst && xq_wr && (x_selw ? xd_wr : xai_rv);
  assign xq_wd    = x_selw ? {1'b1, ~&x_wstb_full, 25'(xwi_rd[24:0] + x_woff)}
                           : {2'b00, 25'(xai_rd[24:0] + x_roff)};
  assign x_wready = !x_crst && xwi_rv && (!x_wpush || (xq_wr && xd_wr));
  assign xd_wv    = x_wvalid && x_wready && x_wpush;
  assign xd_wd    = {~x_wstb_full, x_wdata_full};
  assign xai_rr   = xq_wv && !x_selw && x_rlastb;
  assign xwi_rr   = x_wvalid && x_wready && x_wlast;
  always_ff @(posedge xclk) begin
    if (x_crst) begin
      x_roff <= '0; x_woff <= '0; x_wlane_r <= '0; x_wfirst <= 1'b1; x_wbuf <= '0; x_wstb <= '0;
    end else begin
      if (xq_wv && !x_selw) x_roff <= x_rlastb ? '0 : x_roff + 1'b1;
      if (x_wvalid && x_wready) begin
        x_wfirst  <= x_wlast;
        x_wlane_r <= x_wlane + 1'b1;
        if (x_wpush) begin
          x_woff <= x_wlast ? '0 : x_woff + 1'b1;
          x_wbuf <= '0; x_wstb <= '0;
        end else begin
          x_wbuf <= x_wdata_full; x_wstb <= x_wstb_full;
        end
      end
    end
  end

  // R: lanes of the returned 64-byte beats
  logic [7:0] x_rcnt;
  logic [1:0] x_rlane_r, x_rlane;
  logic       x_rfirst;
  assign x_rlane  = x_rfirst ? xrq_rd[1:0] : x_rlane_r;
  assign x_rvalid = xr_rv && xrq_rv;
  assign x_rdata  = xr_rd[x_rlane * 128 +: 128];
  assign x_rid    = xrq_rd[XRQ-1 -: XIDW];
  assign x_rlast  = x_rcnt == xrq_rd[9:2];
  assign x_rresp  = 2'b00;
  assign xr_rr    = x_rvalid && x_rready && (x_rlane == 2'd3 || x_rlast);
  assign xrq_rr   = x_rvalid && x_rready && x_rlast;
  always_ff @(posedge xclk) begin
    if (x_crst) begin x_rcnt <= '0; x_rlane_r <= '0; x_rfirst <= 1'b1; end
    else if (x_rvalid && x_rready) begin
      x_rcnt    <= x_rlast ? '0 : x_rcnt + 1'b1;
      x_rlane_r <= x_rlane + 1'b1;
      x_rfirst  <= x_rlast;
    end
  end

  // B
  logic [CW-1:0] x_wdone, x_wacc_c;
  always_ff @(posedge xclk) begin
    if (x_crst) begin x_wacc_s1 <= '0; x_wacc_s2 <= '0; x_wacc_c <= '0; end
    else begin x_wacc_s1 <= x_wacc_g; x_wacc_s2 <= x_wacc_s1; x_wacc_c <= g2b(x_wacc_s2); end
  end
  assign x_bvalid = xbq_rv && ((x_wacc_c - x_wdone) > CW'(xbq_rd[6:0]));
  assign x_bid    = xbq_rd[XIDW+6 -: XIDW];
  assign x_bresp  = 2'b00;
  assign xbq_rr   = x_bvalid && x_bready;
  always_ff @(posedge xclk) begin
    if (x_crst) x_wdone <= '0;
    else if (x_bvalid && x_bready) x_wdone <= x_wdone + CW'(xbq_rd[6:0]) + 1'b1;
  end

  // ================================================================ MIG side (uclk)
  // Read credits (registered, one beat of margin: at most one read issues per cycle): a master
  // may issue a read while its reads in flight plus its FIFO's entries leave room for one more.
  logic a_rok, x_rok;
  always_ff @(posedge uclk) begin
    a_rok <= (a_out + ar_used) <= AOW'(ARD - 2);
    x_rok <= (x_out + xr_used) <= XOW'(XRD - 2);
  end

  logic a_ok, x_ok, pick_x, cur_x, go, we;
  logic [5:0] run;
  assign a_ok = aq_rv && (aq_rd[QW-1] ? ad_rv : a_rok);
  assign x_ok = xq_rv && (xq_rd[QW-1] ? xd_rv : x_rok);
  always_comb begin
    if (!cur_x) pick_x = x_ok && (!a_ok || run >= 6'(RUN));
    else        pick_x = !a_ok || (x_ok && run < 6'(RUN));
  end
  assign we           = pick_x ? xq_rd[QW-1] : aq_rd[QW-1];
  assign go           = (pick_x ? x_ok : a_ok) && app_rdy && (!we || app_wdf_rdy);
  assign app_en       = go;
  // write: 000, ECC write-bytes (read-modify-write, any byte kept): 011, read: 001
  assign app_cmd      = !we ? 3'b001 : (pick_x ? xq_rd[QW-2] : aq_rd[QW-2]) ? 3'b011 : 3'b000;
  assign app_addr     = {1'b0, (pick_x ? xq_rd[24:0] : aq_rd[24:0]), 3'b000};
  assign app_wdf_wren = go && we;
  assign app_wdf_end  = go && we;
  assign app_wdf_data = pick_x ? xd_rd[511:0] : ad_rd[511:0];
  assign app_wdf_mask = pick_x ? xd_rd[575:512] : ad_rd[575:512];
  assign aq_rr        = go && !pick_x;
  assign xq_rr        = go && pick_x;
  assign ad_rr        = go && !pick_x && we;
  assign xd_rr        = go && pick_x && we;

  // each read's master, in command order; its data back to that master (dropped under a hold)
  logic tg_wr, tg_rv, tg_rd;
  otpu_sfifo #(.W(1), .DEPTH(128)) u_tag (.clk(uclk), .rst(urst), .wvalid(go && !we), .wready(tg_wr),
    .wdata(pick_x), .rvalid(tg_rv), .rready(app_rd_data_valid), .rdata(tg_rd));
  assign ar_wv = app_rd_data_valid && !tg_rd && !a_hold;
  assign xr_wv = app_rd_data_valid && tg_rd && !x_hold;
  assign ar_wd = app_rd_data;
  assign xr_wd = app_rd_data;

  always_ff @(posedge uclk) begin
    if (urst) begin
      cur_x <= 1'b0; run <= '0; a_out <= '0; x_out <= '0;
    end else begin
      if (go) begin
        run   <= (pick_x != cur_x) ? 6'd1 : (run == 6'd63) ? run : run + 1'b1;
        cur_x <= pick_x;
      end
      a_out <= a_out + AOW'(go && !we && !pick_x) - AOW'(app_rd_data_valid && !tg_rd);
      x_out <= x_out + XOW'(go && !we && pick_x) - XOW'(app_rd_data_valid && tg_rd);
    end
    if (a_hold) a_wacc <= '0; else if (go && we && !pick_x) a_wacc <= a_wacc + 1'b1;
    if (x_hold) x_wacc <= '0; else if (go && we && pick_x) x_wacc <= x_wacc + 1'b1;
    a_wacc_g <= b2g(a_wacc);
    x_wacc_g <= b2g(x_wacc);
  end

`ifndef SYNTHESIS
  // read data always finds room (the credits) and a tag
  always_ff @(posedge uclk) if (!urst && app_rd_data_valid) begin
    if (!tg_rv) $error("otpu_mig_ch: read data without a tag");
    if (ar_wv && !ar_wr) $error("otpu_mig_ch: accelerator read-data FIFO full");
    if (xr_wv && !xr_wr) $error("otpu_mig_ch: XDMA read-data FIFO full");
  end
  always_ff @(posedge uclk) if (!urst && go && !we && !tg_wr) $error("otpu_mig_ch: tag FIFO full");
`endif
endmodule

// Synchronous FIFO, first-word fall-through, distributed RAM (the MIG bridge's queues).
module otpu_sfifo #(
  parameter int W = 8,
  parameter int DEPTH = 16
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         wvalid,
  output logic         wready,
  input  logic [W-1:0] wdata,
  output logic         rvalid,
  input  logic         rready,
  output logic [W-1:0] rdata
);
  localparam int AW = $clog2(DEPTH);
  (* ram_style = "distributed" *) logic [W-1:0] mem [DEPTH];
  logic [AW:0] wp, rp;
  assign wready = (wp - rp) != (AW + 1)'(DEPTH);
  assign rvalid = wp != rp;
  assign rdata  = mem[rp[AW-1:0]];
  always_ff @(posedge clk) begin
    if (rst) begin wp <= '0; rp <= '0; end
    else begin
      if (wvalid && wready) wp <= wp + 1'b1;
      if (rvalid && rready) rp <= rp + 1'b1;
    end
  end
  always_ff @(posedge clk) if (wvalid && wready) mem[wp[AW-1:0]] <= wdata;
endmodule
