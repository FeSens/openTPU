// Unit test of the MIG native path (rtl/boards/ypcb-00338: otpu_mig_ch, otpu_axi_split2,
// otpu_afifo): two accelerator masters (512-bit, core clock), one per channel, and an XDMA
// master (128-bit, axi_aclk) through the split onto both channels, each channel a native MIG
// model (otpu_mig_model, ui_clk). Three unrelated clocks (+cp= / +up= / +xp=: half periods).
// The masters issue random INCR bursts (random lengths, IDs and strobes, XDMA's 16-byte beats
// from any 16-byte offset) over their own windows, with random valid gaps and ready
// backpressure, and check every read beat against a byte shadow of their window. A master
// reads only beats with no write in flight and writes only beats with no access in flight, so
// the expected data is exact (reads and writes are not ordered against each other in AXI).
// +xreset=N / +areset=N: reset the XDMA master (and the split, and the bridges' XDMA sides) /
// the accelerator side at cycle N of its clock, for 40 cycles, mid-traffic; the bytes of the
// writes then in flight become unknown (not checked) until written again. +ntx=N transactions
// per master; +mstall=P: the masters' bready / rready low P percent of the cycles (default 30).
// Throughput: +seq=1 (back-to-back LMAX-beat bursts through the window), +wpct=P (percent of
// writes, default 50), +gapw=P (W valid gaps, default 20); each master prints its data beats per
// cycle of its clock. Prints "PASS" or the first mismatches.
module tb_mig;
  int cp = 414, up = 375, xp = 400;        // half periods: 120.8 / 133.3 / 125 MHz
  logic clk = 1'b0, uclk = 1'b0, xclk = 1'b0;
  initial begin
    int seed;
    void'($value$plusargs("cp=%d", cp));
    void'($value$plusargs("up=%d", up));
    void'($value$plusargs("xp=%d", xp));
    if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));
  end
  always #(cp) clk = ~clk;
  always #(up) uclk = ~uclk;
  always #(xp) xclk = ~xclk;

  localparam int BEATS = 1 << 15;                    // 2 MiB per channel model
  localparam logic [31:0] AWIN = 32'h0000_0000, XWIN = 32'h0010_0000, SPAN = 32'h8000;

  // resets
  logic rst = 1'b1, urst = 1'b1, xrst = 1'b1;
  longint ccyc = 0, xcyc = 0, ucyc = 0;
  longint areset = -1, xreset = -1;
  initial begin
    void'($value$plusargs("areset=%d", areset));
    void'($value$plusargs("xreset=%d", xreset));
  end
  always @(posedge uclk) begin ucyc <= ucyc + 1; if (ucyc == 20) urst <= 1'b0; end
  always @(posedge clk) begin
    ccyc <= ccyc + 1;
    if (ccyc == 30) rst <= 1'b0;
    if (areset > 0 && ccyc == areset) rst <= 1'b1;
    if (areset > 0 && ccyc == areset + 40) rst <= 1'b0;
  end
  always @(posedge xclk) begin
    xcyc <= xcyc + 1;
    if (xcyc == 40) xrst <= 1'b0;
    if (xreset > 0 && xcyc == xreset) xrst <= 1'b1;
    if (xreset > 0 && xcyc == xreset + 40) xrst <= 1'b0;
  end

  // ---- accelerator masters (one per channel) and the bridges
  logic [1:0] awv, awr, wv, wr, wl, bv, br, arv, arr, rv, rr, rl;
  logic [1:0][0:0] awi, bi, ari, ri;
  logic [1:0][31:0] awa, ara;
  logic [1:0][7:0] awl, arl;
  logic [1:0][511:0] wd, rd;
  logic [1:0][63:0] ws;
  logic [1:0][1:0] bre, rre;
  logic [1:0] adone, abad;
  logic xbad;
  // XDMA master -> split -> bridges
  logic xawv, xawr, xwv, xwr, xwl, xbv, xbr, xarv, xarr, xrv, xrr, xrl, xdone;
  logic [3:0] xawi, xbi, xari, xri;
  logic [31:0] xawa, xara;
  logic [7:0] xawl, xarl;
  logic [127:0] xwd, xrd;
  logic [15:0] xws;
  logic [1:0] xbre, xrre;
  logic [1:0] cawv, cawr, cwv, cwr, cbv, cbr, carv, carr, crv, crr, crl;
  logic [1:0][3:0] cbi, cri;
  logic [1:0][1:0] cbre, crre;
  logic [1:0][127:0] crd;
  // MIG native
  logic [1:0][28:0] app_addr;
  logic [1:0][2:0] app_cmd;
  logic [1:0] app_en, app_rdy, app_wdf_wren, app_wdf_end, app_wdf_rdy, app_rd_data_valid;
  logic [1:0][511:0] app_wdf_data, app_rd_data;
  logic [1:0][63:0] app_wdf_mask;
  logic dump = 1'b0;

  for (genvar c = 0; c < 2; c++) begin : g_ch
    tb_mig_master #(.BYTES(64), .IDW(1), .NWIN(1), .WIN0(AWIN | (32'(c) << 31)), .SPAN(SPAN),
                    .LMAX(32), .NAME(c ? "acc1" : "acc0")) u_acc (
      .clk, .rst,
      .awvalid(awv[c]), .awready(awr[c]), .awid(awi[c]), .awaddr(awa[c]), .awlen(awl[c]),
      .wvalid(wv[c]), .wready(wr[c]), .wdata(wd[c]), .wstrb(ws[c]), .wlast(wl[c]),
      .bvalid(bv[c]), .bready(br[c]), .bid(bi[c]), .bresp(bre[c]),
      .arvalid(arv[c]), .arready(arr[c]), .arid(ari[c]), .araddr(ara[c]), .arlen(arl[c]),
      .rvalid(rv[c]), .rready(rr[c]), .rid(ri[c]), .rdata(rd[c]), .rresp(rre[c]), .rlast(rl[c]),
      .done(adone[c]), .bad(abad[c]));

    otpu_mig_ch #(.XIDW(4)) u_ch (
      .clk, .rst,
      .s_awvalid(awv[c]), .s_awready(awr[c]), .s_awid(awi[c]), .s_awaddr(awa[c]), .s_awlen(awl[c]),
      .s_wvalid(wv[c]), .s_wready(wr[c]), .s_wdata(wd[c]), .s_wstrb(ws[c]), .s_wlast(wl[c]),
      .s_bvalid(bv[c]), .s_bready(br[c]), .s_bid(bi[c]), .s_bresp(bre[c]),
      .s_arvalid(arv[c]), .s_arready(arr[c]), .s_arid(ari[c]), .s_araddr(ara[c]), .s_arlen(arl[c]),
      .s_rvalid(rv[c]), .s_rready(rr[c]), .s_rid(ri[c]), .s_rdata(rd[c]), .s_rresp(rre[c]), .s_rlast(rl[c]),
      .xclk, .xrst,
      .x_awvalid(cawv[c]), .x_awready(cawr[c]), .x_awid(xawi), .x_awaddr(xawa), .x_awlen(xawl),
      .x_wvalid(cwv[c]), .x_wready(cwr[c]), .x_wdata(xwd), .x_wstrb(xws), .x_wlast(xwl),
      .x_bvalid(cbv[c]), .x_bready(cbr[c]), .x_bid(cbi[c]), .x_bresp(cbre[c]),
      .x_arvalid(carv[c]), .x_arready(carr[c]), .x_arid(xari), .x_araddr(xara), .x_arlen(xarl),
      .x_rvalid(crv[c]), .x_rready(crr[c]), .x_rid(cri[c]), .x_rdata(crd[c]), .x_rresp(crre[c]),
      .x_rlast(crl[c]),
      .uclk, .urst,
      .app_addr(app_addr[c]), .app_cmd(app_cmd[c]), .app_en(app_en[c]), .app_rdy(app_rdy[c]),
      .app_wdf_data(app_wdf_data[c]), .app_wdf_mask(app_wdf_mask[c]), .app_wdf_wren(app_wdf_wren[c]),
      .app_wdf_end(app_wdf_end[c]), .app_wdf_rdy(app_wdf_rdy[c]), .app_rd_data(app_rd_data[c]),
      .app_rd_data_valid(app_rd_data_valid[c]));

    otpu_mig_model #(.BEATS(BEATS), .CH(c)) u_mig (
      .clk(uclk), .rst(urst),
      .app_addr(app_addr[c]), .app_cmd(app_cmd[c]), .app_en(app_en[c]), .app_rdy(app_rdy[c]),
      .app_wdf_data(app_wdf_data[c]), .app_wdf_mask(app_wdf_mask[c]), .app_wdf_wren(app_wdf_wren[c]),
      .app_wdf_end(app_wdf_end[c]), .app_wdf_rdy(app_wdf_rdy[c]), .app_rd_data(app_rd_data[c]),
      .app_rd_data_valid(app_rd_data_valid[c]), .dump);
  end

  tb_mig_master #(.BYTES(16), .IDW(4), .NWIN(2), .WIN0(XWIN), .WIN1(XWIN | 32'h8000_0000),
                  .SPAN(SPAN), .LMAX(64), .NAME("xdma")) u_xdma (
    .clk(xclk), .rst(xrst),
    .awvalid(xawv), .awready(xawr), .awid(xawi), .awaddr(xawa), .awlen(xawl),
    .wvalid(xwv), .wready(xwr), .wdata(xwd), .wstrb(xws), .wlast(xwl),
    .bvalid(xbv), .bready(xbr), .bid(xbi), .bresp(xbre),
    .arvalid(xarv), .arready(xarr), .arid(xari), .araddr(xara), .arlen(xarl),
    .rvalid(xrv), .rready(xrr), .rid(xri), .rdata(xrd), .rresp(xrre), .rlast(xrl),
    .done(xdone), .bad(xbad));

  otpu_axi_split2 #(.IDW(4), .DW(128)) u_split (
    .clk(xclk), .rst(xrst),
    .s_awvalid(xawv), .s_awready(xawr), .s_awid(xawi), .s_awaddr(xawa), .s_awlen(xawl),
    .s_wvalid(xwv), .s_wready(xwr), .s_wdata(xwd), .s_wstrb(xws), .s_wlast(xwl),
    .s_bvalid(xbv), .s_bready(xbr), .s_bid(xbi), .s_bresp(xbre),
    .s_arvalid(xarv), .s_arready(xarr), .s_arid(xari), .s_araddr(xara), .s_arlen(xarl),
    .s_rvalid(xrv), .s_rready(xrr), .s_rid(xri), .s_rdata(xrd), .s_rresp(xrre), .s_rlast(xrl),
    .m_awvalid(cawv), .m_awready(cawr), .m_wvalid(cwv), .m_wready(cwr),
    .m_bvalid(cbv), .m_bready(cbr), .m_bid(cbi), .m_bresp(cbre),
    .m_arvalid(carv), .m_arready(carr),
    .m_rvalid(crv), .m_rready(crr), .m_rid(cri), .m_rdata(crd), .m_rresp(crre), .m_rlast(crl));

  // the end: every master done
  longint tmax = 2000000;
  initial void'($value$plusargs("tmax=%d", tmax));
  always @(posedge clk) begin
    if (&adone && xdone) begin
      dump <= 1'b1;
      repeat (4) @(posedge clk);
      $display("%s cycles=%0d", (|abad || xbad) ? "FAIL" : "PASS", ccyc);
      $finish;
    end
    if (ccyc > tmax) begin
      $display("TIMEOUT acc done %b xdma done %b", adone, xdone);
      $finish;
    end
  end
endmodule

// A random AXI4 master over its windows (see tb_mig): INCR bursts of BYTES-byte beats.
module tb_mig_master #(
  parameter int BYTES = 64,
  parameter int IDW = 1,
  parameter int NWIN = 1,
  parameter logic [31:0] WIN0 = 32'h0,
  parameter logic [31:0] WIN1 = 32'h0,
  parameter logic [31:0] SPAN = 32'h8000,
  parameter int LMAX = 32,
  parameter int OUTS = 8,                  // bursts in flight per direction
  parameter string NAME = "m"
) (
  input  logic                 clk,
  input  logic                 rst,
  output logic                 awvalid,
  input  logic                 awready,
  output logic [IDW-1:0]       awid,
  output logic [31:0]          awaddr,
  output logic [7:0]           awlen,
  output logic                 wvalid,
  input  logic                 wready,
  output logic [BYTES*8-1:0]   wdata,
  output logic [BYTES-1:0]     wstrb,
  output logic                 wlast,
  input  logic                 bvalid,
  output logic                 bready,
  input  logic [IDW-1:0]       bid,
  input  logic [1:0]           bresp,
  output logic                 arvalid,
  input  logic                 arready,
  output logic [IDW-1:0]       arid,
  output logic [31:0]          araddr,
  output logic [7:0]           arlen,
  input  logic                 rvalid,
  output logic                 rready,
  input  logic [IDW-1:0]       rid,
  input  logic [BYTES*8-1:0]   rdata,
  input  logic [1:0]           rresp,
  input  logic                 rlast,
  output logic                 done,
  output logic                 bad
);
  localparam int NB = NWIN * SPAN;          // shadow bytes
  logic [7:0] sh [NB];
  bit         unk [NB];
  int         wp [NB / 64], rp [NB / 64];   // writes / reads in flight per 64-byte beat
  typedef struct { logic [BYTES*8-1:0] d; logic [BYTES-1:0] s; bit last; } w_t;
  typedef struct { int idx; logic [IDW-1:0] id; int nb; bit last; } r_t;     // one R beat
  typedef struct { int b0; int nb; logic [IDW-1:0] id; } x_t;                 // one burst's 64-byte beats
  w_t wq [$];
  r_t rq [$];
  x_t bq [$], rdq [$];
  int ntx = 400, issued = 0, nerr = 0;
  longint nrb = 0, nwb = 0;
  int gapw = 20, mstall = 30, wpct = 50, seq = 0;
  logic [31:0] nexto = 0;                   // +seq: the next burst's offset
  longint c0 = -1, c1 = 0, cyc = 0;         // first and last data cycles

  function automatic int idx_of(input logic [31:0] a);
    if (NWIN == 2 && a[31] != WIN0[31]) return int'(SPAN + (a - WIN1));
    return int'(a - WIN0);
  endfunction

  initial begin
    void'($value$plusargs("ntx=%d", ntx));
    void'($value$plusargs({NAME, "_ntx=%d"}, ntx));
    void'($value$plusargs("mstall=%d", mstall));
    void'($value$plusargs("wpct=%d", wpct));
    void'($value$plusargs("seq=%d", seq));
    void'($value$plusargs("gapw=%d", gapw));
    for (int i = 0; i < NB; i++) begin sh[i] = 8'h00; unk[i] = 1'b0; end
    for (int i = 0; i < NB / 64; i++) begin wp[i] = 0; rp[i] = 0; end
  end

  always @(posedge clk) begin
    cyc++;
    if ((rvalid && rready) || (wvalid && wready)) begin if (c0 < 0) c0 = cyc; c1 = cyc; end
    if (rst) begin
      // writes in flight may or may not land: their beats become unknown
      foreach (bq[j]) for (int b = bq[j].b0; b < bq[j].b0 + bq[j].nb; b++)
        for (int k = 0; k < 64; k++) unk[64 * b + k] = 1'b1;
      for (int i = 0; i < NB / 64; i++) begin wp[i] = 0; rp[i] = 0; end
      wq.delete(); rq.delete(); bq.delete(); rdq.delete();
      awvalid <= 1'b0; wvalid <= 1'b0; arvalid <= 1'b0; bready <= 1'b0; rready <= 1'b0;
    end else begin
      bit whs;
      // ---- responses
      if (bvalid && bready) begin
        if (bq.size() == 0) begin $display("ERROR %s: B without a write", NAME); nerr++; end
        else begin
          if (bid != bq[0].id || bresp != 0) begin
            $display("ERROR %s: B id %h resp %h, expected id %h", NAME, bid, bresp, bq[0].id); nerr++;
          end
          for (int b = bq[0].b0; b < bq[0].b0 + bq[0].nb; b++) wp[b]--;
          void'(bq.pop_front());
        end
      end
      if (rvalid && rready) begin
        if (rq.size() == 0) begin $display("ERROR %s: R without a read", NAME); nerr++; end
        else begin
          for (int k = 0; k < BYTES; k++)
            if (!unk[rq[0].idx + k] && rdata[8 * k +: 8] !== sh[rq[0].idx + k]) begin
              if (nerr < 20)
                $display("ERROR %s: read byte %0d: %h, expected %h", NAME, rq[0].idx + k,
                         rdata[8 * k +: 8], sh[rq[0].idx + k]);
              nerr++;
            end
          if (rid != rq[0].id || rlast != rq[0].last || rresp != 0) begin
            $display("ERROR %s: R id %h last %b resp %h, expected id %h last %b", NAME, rid, rlast,
                     rresp, rq[0].id, rq[0].last);
            nerr++;
          end
          nrb++;
          if (rq[0].last) begin
            for (int b = rdq[0].b0; b < rdq[0].b0 + rdq[0].nb; b++) rp[b]--;
            void'(rdq.pop_front());
          end
          void'(rq.pop_front());
        end
      end
      bready <= ($urandom % 100) >= mstall;
      rready <= ($urandom % 100) >= mstall;
      // ---- W
      whs = wvalid && wready;
      if (whs) begin void'(wq.pop_front()); nwb++; end
      if (!wvalid || whs) begin
        if (wq.size() != 0 && ($urandom % 100) >= gapw) begin
          wvalid <= 1'b1; wdata <= wq[0].d; wstrb <= wq[0].s; wlast <= wq[0].last;
        end else wvalid <= 1'b0;
      end
      // ---- AW / AR: a new burst when the channel is free
      if (awvalid && awready) awvalid <= 1'b0;
      if (arvalid && arready) arvalid <= 1'b0;
      if (issued < ntx && ($urandom % 100) < 50) begin
        bit wr_;
        int w, n, i0, b0, b1;
        logic [31:0] off, a;
        bit ok;
        wr_ = ($urandom % 100) < wpct;
        w = (NWIN == 2) ? $urandom % 2 : 0;
        off = seq ? nexto : ($urandom % (SPAN / BYTES)) * BYTES;
        a = (w ? WIN1 : WIN0) + off;
        n = seq ? LMAX : 1 + $urandom % LMAX;
        if (n * BYTES > 4096 - (a & 4095)) n = (4096 - (a & 4095)) / BYTES;
        if (n * BYTES > SPAN - off) n = (SPAN - off) / BYTES;
        i0 = idx_of(a);
        b0 = i0 / 64; b1 = (i0 + n * BYTES - 1) / 64;
        ok = wr_ ? (!(awvalid && !awready) && bq.size() < OUTS) : (!(arvalid && !arready) && rdq.size() < OUTS);
        for (int b = b0; b <= b1 && ok; b++) if (wp[b] != 0 || (wr_ && rp[b] != 0)) ok = 1'b0;
        if (ok) begin
          logic [IDW-1:0] id;
          nexto = (off + n * BYTES) % SPAN;
          id = IDW'($urandom);
          issued++;
          if (wr_) begin
            bit full;
            full = ($urandom % 100) < 70;
            for (int j = 0; j < n; j++) begin
              w_t x;
              for (int k = 0; k < BYTES; k++) begin
                x.d[8 * k +: 8] = 8'($urandom);
                x.s[k] = full || ($urandom % 2);
                if (x.s[k]) begin sh[i0 + j * BYTES + k] = x.d[8 * k +: 8]; unk[i0 + j * BYTES + k] = 1'b0; end
              end
              x.last = j == n - 1;
              wq.push_back(x);
            end
            for (int b = b0; b <= b1; b++) wp[b]++;
            bq.push_back('{b0, b1 - b0 + 1, id});
            awvalid <= 1'b1; awaddr <= a; awlen <= 8'(n - 1); awid <= id;
          end else begin
            for (int j = 0; j < n; j++) rq.push_back('{i0 + j * BYTES, id, 0, j == n - 1});
            for (int b = b0; b <= b1; b++) rp[b]++;
            rdq.push_back('{b0, b1 - b0 + 1, id});
            arvalid <= 1'b1; araddr <= a; arlen <= 8'(n - 1); arid <= id;
          end
        end
      end
    end
  end
  assign done = (issued >= ntx && bq.size() == 0 && rdq.size() == 0 && wq.size() == 0 && !awvalid &&
                 !arvalid) || nerr != 0;
  assign bad  = nerr != 0;

  final begin
    $display("%s: %0d bursts, %0d read beats, %0d write beats, %0d errors, %.3f beats per cycle", NAME,
             issued, nrb, nwb, nerr, real'(nrb + nwb) / real'(c1 - c0 + 1));
  end
endmodule
