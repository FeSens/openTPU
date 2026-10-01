// Unit test of the memory channel path (rtl/boards/ypcb-00338: otpu_mem_ch, otpu_dma_split,
// otpu_afifo) in front of the controllers: two random native masters (the accelerator's side,
// core clock), one per channel, and an XDMA master (128-bit AXI, axi_aclk) through the split onto
// both channels (XREG: with its register slices, as at PCIe Gen2); each channel's controller is a model of LiteDRAM's two native ports (sys; a
// port per bank parity, the ports unordered against each other), which takes whole beats only
// (otpu_mem_ch's read-modify-write). Three unrelated clocks (+cp= / +up= / +xp=: half periods).
//
// The native master runs a program of random runs of 1 to 32 beats over its window (reads or
// writes, +wpct=P percent writes; +ppct=P percent of the write beats partial, with one byte, all
// but one, a range, none or random bytes written; +raw=P percent of the runs read back the last
// write run at once). Commands and write data go out independently (random gaps: +gapc=P,
// +gapw=P), so a write's data comes before or after its command. It keeps room for its reads (a
// buffer of +rbuf=N beats, drained with +mstall=P percent stalls) and checks every read beat
// against a byte shadow as of the read's place in the program, since one master's commands run
// in order. It checks that n_wdone counts every write exactly once, never one whose command or
// data it has not yet given.
// The XDMA master issues random INCR bursts (random lengths, IDs and strobes, +xfull=P percent
// of bursts with whole strobes; 16-byte beats from any 16-byte offset) over its own windows, with
// random valid gaps and ready backpressure (+mstall), and checks every read beat against a byte
// shadow; it reads only beats with no write in flight and writes only beats with no access in
// flight (AXI does not order them).
// Across masters (+psh=P percent of shared operations): each accelerator writes a shared region
// once, beat by beat (whole, or two complementary partial writes), and publishes a beat when
// n_wdone has counted it; XDMA reads each published beat at once and checks it. XDMA writes
// another region, publishes on B, and the accelerator reads and checks. The data is a hash of the
// beat, so a read that passes a write shows the old zeros. In a split region (+psp=P percent)
// each accelerator owns bytes 0-31 of every beat and XDMA bytes 32-63: both write their halves
// with partial writes at random and read them back, so a read-modify-write that let the other
// master's write in between its read and its write would undo that write.
// +xreset=N / +areset=N: reset the XDMA master (and the split, and the bridges' XDMA sides) / the
// accelerators at cycle N of their clock, for 40 cycles (+xrlen=L / +arlen=L: L cycles),
// mid-traffic (+xrep=P / +arep=P: again every P cycles); the bytes of the writes then in flight become unknown (not checked) until
// written again, and unpublished shared beats are written again. +ntx=N runs or bursts per master (+acc0_ntx, +acc1_ntx, +xdma_ntx).
// Throughput: +seq=1 (back-to-back 32-beat runs / 64-beat bursts through the window, no shared
// operations); each master prints its data beats per cycle of its clock, from its first command
// to its last read beat or write counted. Controller models: +axi_stall, +axi_lat, +ldn_busy.
// The bridges' error bits (n_err) must stay clear; with +ldn_dual=N (both channel models return
// a beat on both ports once, a broken controller; run with +verilator+error+limit+N so the
// bridge's own check does not stop it) both must set, and the run ends there.
// Prints "PASS" or the first mismatches.
package tb_memch_pkg;
  // a shared beat's data: a hash of (channel, direction, beat)
  function automatic logic [511:0] pat(input int c, input int dir, input int i);
    logic [511:0] d;
    for (int k = 0; k < 16; k++) begin
      logic [31:0] h;
      h = 32'(c) * 32'h9E3779B1 ^ 32'(dir) * 32'h85EBCA77 ^ 32'(i) * 32'hC2B2AE3D ^ 32'(k + 1) * 32'h27D4EB2F;
      h = h ^ (h >> 15); h = h * 32'h2C1B3C6D; h = h ^ (h >> 12); h = h * 32'h297A2D39; h = h ^ (h >> 15);
      d[32 * k +: 32] = h;
    end
    return d;
  endfunction
endpackage

module tb_memch #(
  parameter int ARD = 64,
  parameter int XRD = 16,
  parameter bit XREG = 1'b0
);
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

  // per channel (beats): the accelerator's window, XDMA's (bytes 0x10_0000..), the shared regions
  localparam int BEATS = 1 << 15;                    // 2 MiB per channel model
  localparam int NB = 512, A2X = 'h6000, X2A = 'h6100, NS = 256, SPL = 'h6200, NSP = 64;
  localparam logic [31:0] XWIN = 32'h0010_0000, SPAN = 32'h8000;

  // resets
  logic rst = 1'b1, urst = 1'b1, xrst = 1'b1;
  longint ccyc = 0, xcyc = 0, ucyc = 0;
  longint areset = -1, xreset = -1, arep = 0, xrep = 0, arlen = 40, xrlen = 40;
  initial begin
    void'($value$plusargs("areset=%d", areset));
    void'($value$plusargs("xreset=%d", xreset));
    void'($value$plusargs("arep=%d", arep));
    void'($value$plusargs("xrep=%d", xrep));
    void'($value$plusargs("arlen=%d", arlen));
    void'($value$plusargs("xrlen=%d", xrlen));
  end
  // cycle c of a reset that starts at s and repeats every p cycles (p = 0: once)
  function automatic bit at(input longint c, input longint s, input longint p);
    return s > 0 && c >= s && (p > 0 ? (c - s) % p == 0 : c == s);
  endfunction
  always @(posedge uclk) begin ucyc <= ucyc + 1; if (ucyc == 20) urst <= 1'b0; end
  always @(posedge clk) begin
    ccyc <= ccyc + 1;
    if (ccyc == 30) rst <= 1'b0;
    if (at(ccyc, areset, arep)) rst <= 1'b1;
    if (at(ccyc, areset + arlen, arep)) rst <= 1'b0;
  end
  always @(posedge xclk) begin
    xcyc <= xcyc + 1;
    if (xcyc == 40) xrst <= 1'b0;
    if (at(xcyc, xreset, xrep)) xrst <= 1'b1;
    if (at(xcyc, xreset + xrlen, xrep)) xrst <= 1'b0;
  end

  // ---- accelerator masters (one per channel), the bridges and the controllers
  logic [1:0] ncv, ncr, ncwe, nwv, nwr, nrv;
  logic [1:0][24:0] nca;
  logic [1:0][511:0] nwd, nrd;
  logic [1:0][63:0] nwm;
  logic [1:0][15:0] nwdone;
  logic [1:0] nerr;
  logic [1:0][31:0] a2x_pub, x2a_pub;
  logic [1:0] adone, abad;
  logic [1:0][1:0] ccv, ccr, ccwe, cwv, cwr, crv;     // [channel][port]
  logic [1:0][1:0][24:0] cca;
  logic [1:0][1:0][511:0] cwd, crd;
  logic [1:0][1:0][63:0] cwe;
  // XDMA master -> split -> bridges
  logic xawv, xawr, xwv, xwr, xwl, xbv, xbr, xarv, xarr, xrv, xrr, xrl, xdone, xbad;
  logic [3:0] xawi, xbi, xari, xri;
  logic [31:0] xawa, xara;
  logic [7:0] xawl, xarl;
  logic [127:0] xwd, xrd;
  logic [15:0] xws;
  logic [1:0] xbre, xrre;
  logic [1:0] cawv, cawr, cwvx, cwrx, cwl, cbv, cbr, carv, carr, crvx, crrx, crl;
  logic [1:0][3:0] cawi, cari, cbi, cri;
  logic [1:0][31:0] cawa, cara;
  logic [1:0][7:0] cawl, carl;
  logic [1:0][1:0] cbre, crre;
  logic [1:0][127:0] cwdx, crdx;
  logic [1:0][15:0] cws;
  logic dump = 1'b0;

  for (genvar c = 0; c < 2; c++) begin : g_ch
    tb_memch_nat #(.CH(c), .NB(NB), .A2X(A2X), .X2A(X2A), .NS(NS), .SPL(SPL), .NSP(NSP), .NAME(c ? "acc1" : "acc0")) u_acc (
      .clk, .rst,
      .n_cvalid(ncv[c]), .n_cready(ncr[c]), .n_cwe(ncwe[c]), .n_caddr(nca[c]),
      .n_wvalid(nwv[c]), .n_wready(nwr[c]), .n_wdata(nwd[c]), .n_wmask(nwm[c]),
      .n_rvalid(nrv[c]), .n_rdata(nrd[c]), .n_wdone(nwdone[c]),
      .a2x_pub(a2x_pub[c]), .x2a_pub(x2a_pub[c]), .done(adone[c]), .bad(abad[c]));

    otpu_mem_ch #(.XIDW(4), .ARD(ARD), .XRD(XRD)) u_ch (
      .clk, .rst,
      .n_cvalid(ncv[c]), .n_cready(ncr[c]), .n_cwe(ncwe[c]), .n_caddr(nca[c]),
      .n_wvalid(nwv[c]), .n_wready(nwr[c]), .n_wdata(nwd[c]), .n_wmask(nwm[c]),
      .n_rvalid(nrv[c]), .n_rdata(nrd[c]), .n_wdone(nwdone[c]), .n_err(nerr[c]),
      .xclk, .xrst,
      .x_awvalid(cawv[c]), .x_awready(cawr[c]), .x_awid(cawi[c]), .x_awaddr(cawa[c]),
      .x_awlen(cawl[c]), .x_wvalid(cwvx[c]), .x_wready(cwrx[c]), .x_wdata(cwdx[c]),
      .x_wstrb(cws[c]), .x_wlast(cwl[c]),
      .x_bvalid(cbv[c]), .x_bready(cbr[c]), .x_bid(cbi[c]), .x_bresp(cbre[c]),
      .x_arvalid(carv[c]), .x_arready(carr[c]), .x_arid(cari[c]), .x_araddr(cara[c]),
      .x_arlen(carl[c]),
      .x_rvalid(crvx[c]), .x_rready(crrx[c]), .x_rid(cri[c]), .x_rdata(crdx[c]), .x_rresp(crre[c]),
      .x_rlast(crl[c]),
      .uclk, .urst,
      .c_cmd_valid(ccv[c]), .c_cmd_ready(ccr[c]), .c_cmd_we(ccwe[c]), .c_cmd_addr(cca[c]),
      .c_wdata_valid(cwv[c]), .c_wdata_ready(cwr[c]), .c_wdata_data(cwd[c]),
      .c_wdata_we(cwe[c]), .c_rdata_valid(crv[c]), .c_rdata_data(crd[c]));

    otpu_ldn_model #(.BEATS(BEATS), .CH(c)) u_mem (
      .clk(uclk), .rst(urst),
      .c_cmd_valid(ccv[c]), .c_cmd_ready(ccr[c]), .c_cmd_we(ccwe[c]), .c_cmd_addr(cca[c]),
      .c_wdata_valid(cwv[c]), .c_wdata_ready(cwr[c]), .c_wdata_data(cwd[c]),
      .c_wdata_we(cwe[c]), .c_rdata_valid(crv[c]), .c_rdata_data(crd[c]), .dump);
  end

  tb_memch_axi #(.IDW(4), .WIN0(XWIN), .WIN1(XWIN | 32'h8000_0000), .SPAN(SPAN), .LMAX(64),
                 .A2X(A2X), .X2A(X2A), .NS(NS), .SPL(SPL), .NSP(NSP), .NAME("xdma")) u_xdma (
    .clk(xclk), .rst(xrst),
    .awvalid(xawv), .awready(xawr), .awid(xawi), .awaddr(xawa), .awlen(xawl),
    .wvalid(xwv), .wready(xwr), .wdata(xwd), .wstrb(xws), .wlast(xwl),
    .bvalid(xbv), .bready(xbr), .bid(xbi), .bresp(xbre),
    .arvalid(xarv), .arready(xarr), .arid(xari), .araddr(xara), .arlen(xarl),
    .rvalid(xrv), .rready(xrr), .rid(xri), .rdata(xrd), .rresp(xrre), .rlast(xrl),
    .a2x_pub, .x2a_pub, .done(xdone), .bad(xbad));

  otpu_dma_split #(.REG(XREG), .IDW(4), .DW(128)) u_split (
    .clk(xclk), .rst(xrst),
    .s_awvalid(xawv), .s_awready(xawr), .s_awid(xawi), .s_awaddr(xawa), .s_awlen(xawl),
    .s_wvalid(xwv), .s_wready(xwr), .s_wdata(xwd), .s_wstrb(xws), .s_wlast(xwl),
    .s_bvalid(xbv), .s_bready(xbr), .s_bid(xbi), .s_bresp(xbre),
    .s_arvalid(xarv), .s_arready(xarr), .s_arid(xari), .s_araddr(xara), .s_arlen(xarl),
    .s_rvalid(xrv), .s_rready(xrr), .s_rid(xri), .s_rdata(xrd), .s_rresp(xrre), .s_rlast(xrl),
    .m_awvalid(cawv), .m_awready(cawr), .m_awid(cawi), .m_awaddr(cawa), .m_awlen(cawl),
    .m_wvalid(cwvx), .m_wready(cwrx), .m_wdata(cwdx), .m_wstrb(cws), .m_wlast(cwl),
    .m_bvalid(cbv), .m_bready(cbr), .m_bid(cbi), .m_bresp(cbre),
    .m_arvalid(carv), .m_arready(carr), .m_arid(cari), .m_araddr(cara), .m_arlen(carl),
    .m_rvalid(crvx), .m_rready(crrx), .m_rid(cri), .m_rdata(crdx), .m_rresp(crre), .m_rlast(crl));

  // +cov: xclk cycles in which each XDMA-side queue was full (the split's order FIFOs, the bridges'
  // AW / AR queues)
  longint cv_ow = 0, cv_ob = 0, cv_or = 0;
  longint cv_xwi [2] = '{0, 0}, cv_xbq [2] = '{0, 0}, cv_xai [2] = '{0, 0}, cv_xrq [2] = '{0, 0};
  bit cov = 1'b0;
  initial cov = $test$plusargs("cov");
  always @(posedge xclk) if (!xrst) begin
    cv_ow += longint'(!u_split.u_split.u_ow.wready);
    cv_ob += longint'(!u_split.u_split.u_ob.wready);
    cv_or += longint'(!u_split.u_split.u_or.wready);
    cv_xwi[0] += longint'(!g_ch[0].u_ch.xwi_wr);  cv_xwi[1] += longint'(!g_ch[1].u_ch.xwi_wr);
    cv_xbq[0] += longint'(!g_ch[0].u_ch.xbq_wr);  cv_xbq[1] += longint'(!g_ch[1].u_ch.xbq_wr);
    cv_xai[0] += longint'(!g_ch[0].u_ch.xai_wr);  cv_xai[1] += longint'(!g_ch[1].u_ch.xai_wr);
    cv_xrq[0] += longint'(!g_ch[0].u_ch.xrq_wr);  cv_xrq[1] += longint'(!g_ch[1].u_ch.xrq_wr);
  end
  final if (cov)
    $display("cov full cycles: ow %0d ob %0d or %0d | xwi %0d/%0d xbq %0d/%0d xai %0d/%0d xrq %0d/%0d",
             cv_ow, cv_ob, cv_or, cv_xwi[0], cv_xwi[1], cv_xbq[0], cv_xbq[1], cv_xai[0], cv_xai[1],
             cv_xrq[0], cv_xrq[1]);

  // Checks on the XDMA side, every xclk cycle:
  // - each registered-flag queue (otpu_sfifo RO: the split's order FIFOs, the bridges' AW / AR
  //   queues) has rvalid = not empty and wready = not full of its pointers, and never more than
  //   DEPTH entries;
  // - W bursts (WLAST beats taken) never outnumber the AWs taken, at the split's input and at
  //   each bridge, and each bridge's B responses never outnumber its W bursts; at the end all
  //   three counts agree, per bridge and at the split (in bursts since the last XDMA reset).
  int ro_err [11];
  tb_ro_chk #(.DEPTH(16), .NAME("split u_ow")) u_ck0 (.clk(xclk), .rst(xrst),
    .wp(u_split.u_split.u_ow.wp), .rp(u_split.u_split.u_ow.rp), .rvalid(u_split.u_split.u_ow.rvalid),
    .wready(u_split.u_split.u_ow.wready), .nerr(ro_err[0]));
  tb_ro_chk #(.DEPTH(16), .NAME("split u_ob")) u_ck1 (.clk(xclk), .rst(xrst),
    .wp(u_split.u_split.u_ob.wp), .rp(u_split.u_split.u_ob.rp), .rvalid(u_split.u_split.u_ob.rvalid),
    .wready(u_split.u_split.u_ob.wready), .nerr(ro_err[1]));
  tb_ro_chk #(.DEPTH(16), .NAME("split u_or")) u_ck2 (.clk(xclk), .rst(xrst),
    .wp(u_split.u_split.u_or.wp), .rp(u_split.u_split.u_or.rp), .rvalid(u_split.u_split.u_or.rvalid),
    .wready(u_split.u_split.u_or.wready), .nerr(ro_err[2]));
  for (genvar c = 0; c < 2; c++) begin : g_ck
    tb_ro_chk #(.DEPTH(4), .NAME(c ? "ch1 u_xai" : "ch0 u_xai")) u_ckai (.clk(xclk), .rst(g_ch[c].u_ch.x_crst),
      .wp(g_ch[c].u_ch.u_xai.wp), .rp(g_ch[c].u_ch.u_xai.rp), .rvalid(g_ch[c].u_ch.u_xai.rvalid),
      .wready(g_ch[c].u_ch.u_xai.wready), .nerr(ro_err[3 + 4 * c]));
    tb_ro_chk #(.DEPTH(4), .NAME(c ? "ch1 u_xwi" : "ch0 u_xwi")) u_ckwi (.clk(xclk), .rst(g_ch[c].u_ch.x_crst),
      .wp(g_ch[c].u_ch.u_xwi.wp), .rp(g_ch[c].u_ch.u_xwi.rp), .rvalid(g_ch[c].u_ch.u_xwi.rvalid),
      .wready(g_ch[c].u_ch.u_xwi.wready), .nerr(ro_err[4 + 4 * c]));
    tb_ro_chk #(.DEPTH(16), .NAME(c ? "ch1 u_xrq" : "ch0 u_xrq")) u_ckrq (.clk(xclk), .rst(g_ch[c].u_ch.x_crst),
      .wp(g_ch[c].u_ch.u_xrq.wp), .rp(g_ch[c].u_ch.u_xrq.rp), .rvalid(g_ch[c].u_ch.u_xrq.rvalid),
      .wready(g_ch[c].u_ch.u_xrq.wready), .nerr(ro_err[5 + 4 * c]));
    tb_ro_chk #(.DEPTH(16), .NAME(c ? "ch1 u_xbq" : "ch0 u_xbq")) u_ckbq (.clk(xclk), .rst(g_ch[c].u_ch.x_crst),
      .wp(g_ch[c].u_ch.u_xbq.wp), .rp(g_ch[c].u_ch.u_xbq.rp), .rvalid(g_ch[c].u_ch.u_xbq.rvalid),
      .wready(g_ch[c].u_ch.u_xbq.wready), .nerr(ro_err[6 + 4 * c]));
  end
  longint n_aw_i = 0, n_wl_i = 0, n_aw [2] = '{0, 0}, n_wl [2] = '{0, 0}, n_b [2] = '{0, 0};
  int cnt_err = 0;
  always @(posedge xclk) begin
    if (xrst) begin
      n_aw_i = 0; n_wl_i = 0;
      for (int c = 0; c < 2; c++) begin n_aw[c] = 0; n_wl[c] = 0; n_b[c] = 0; end
    end else begin
      n_aw_i += longint'(u_split.i_awvalid && u_split.i_awready);
      n_wl_i += longint'(u_split.i_wvalid && u_split.i_wready && u_split.i_wlast);
      for (int c = 0; c < 2; c++) begin
        n_aw[c] += longint'(cawv[c] && cawr[c]);
        n_wl[c] += longint'(cwvx[c] && cwrx[c] && cwl[c]);
        n_b[c]  += longint'(cbv[c] && cbr[c]);
      end
      if (n_wl_i > n_aw_i || n_wl[0] > n_aw[0] || n_wl[1] > n_aw[1] || n_b[0] > n_wl[0] ||
          n_b[1] > n_wl[1]) begin
        if (cnt_err < 5)
          $display("ERROR counts: split AW %0d WLAST %0d; ch0 AW %0d WLAST %0d B %0d; ch1 AW %0d WLAST %0d B %0d",
                   n_aw_i, n_wl_i, n_aw[0], n_wl[0], n_b[0], n_aw[1], n_wl[1], n_b[1]);
        cnt_err++;
      end
    end
  end
  function automatic bit xchk_bad();
    int e = cnt_err;
    for (int i = 0; i < 11; i++) e += ro_err[i];
    if (n_aw_i != n_wl_i || n_aw[0] != n_wl[0] || n_aw[1] != n_wl[1] || n_b[0] != n_aw[0] ||
        n_b[1] != n_aw[1] || n_aw_i != n_aw[0] + n_aw[1]) begin
      $display("ERROR counts at the end: split AW %0d WLAST %0d; ch0 AW %0d WLAST %0d B %0d; ch1 AW %0d WLAST %0d B %0d",
               n_aw_i, n_wl_i, n_aw[0], n_wl[0], n_b[0], n_aw[1], n_wl[1], n_b[1]);
      e++;
    end
    return e != 0;
  endfunction
  final if (cov) $display("cov counts: split AW %0d WLAST %0d; ch0 AW %0d B %0d; ch1 AW %0d B %0d",
                          n_aw_i, n_wl_i, n_aw[0], n_b[0], n_aw[1], n_b[1]);

  // the end: every master done; or the error bits
  longint tmax = 2000000, dual = 0;
  initial void'($value$plusargs("tmax=%d", tmax));
  initial void'($value$plusargs("ldn_dual=%d", dual));
  always @(posedge clk) begin
    if (dual == 0 && |nerr) begin
      $display("FAIL n_err %b: a bridge saw its controller break the port contract", nerr);
      $finish;
    end
    if (dual != 0 && &nerr) begin
      $display("PASS cycles=%0d: both bridges' n_err set by the double beat", ccyc);
      $finish;
    end
    if (&adone && xdone) begin
      dump <= 1'b1;
      repeat (4) @(posedge clk);
      if (dual != 0) $display("ERROR n_err %b after the double beat", nerr);
      $display("%s cycles=%0d", (|abad || xbad || dual != 0 || xchk_bad()) ? "FAIL" : "PASS", ccyc);
      $finish;
    end
    if (ccyc > tmax) begin
      $display("TIMEOUT acc done %b xdma done %b", adone, xdone);
      $finish;
    end
  end
endmodule

// One otpu_sfifo RO's registered flags against its pointers (tb_memch's XDMA-side checks).
module tb_ro_chk #(
  parameter int DEPTH = 16,
  parameter string NAME = "fifo"
) (
  input  logic                     clk,
  input  logic                     rst,
  input  logic [$clog2(DEPTH):0]   wp,
  input  logic [$clog2(DEPTH):0]   rp,
  input  logic                     rvalid,
  input  logic                     wready,
  output int                       nerr
);
  localparam int AW = $clog2(DEPTH);
  logic [AW:0] used;
  assign used = wp - rp;
  initial nerr = 0;
  always @(posedge clk) if (!rst) begin
    if (rvalid !== (used != 0) || wready !== (used != (AW + 1)'(DEPTH)) || used > (AW + 1)'(DEPTH)) begin
      if (nerr < 5)
        $display("ERROR %s: rvalid %b wready %b with %0d entries (wp %0d rp %0d)", NAME, rvalid,
                 wready, used, wp, rp);
      nerr++;
    end
  end
endmodule

// A random native master (the accelerator's side of otpu_mem_ch; see tb_memch).
module tb_memch_nat #(
  parameter int CH = 0,
  parameter int NB = 512,                  // its window: beats 0 .. NB - 1 of the channel
  parameter int A2X = 0,                   // shared regions: the beats it writes, the ones it reads
  parameter int X2A = 0,
  parameter int NS = 256,
  parameter int SPL = 0,                   // split region: it owns bytes 0-31 of each beat, XDMA 32-63
  parameter int NSP = 64,
  parameter int LMAX = 32,
  parameter string NAME = "acc"
) (
  input  logic         clk,
  input  logic         rst,
  output logic         n_cvalid,
  input  logic         n_cready,
  output logic         n_cwe,
  output logic [24:0]  n_caddr,
  output logic         n_wvalid,
  input  logic         n_wready,
  output logic [511:0] n_wdata,
  output logic [63:0]  n_wmask,
  input  logic         n_rvalid,
  input  logic [511:0] n_rdata,
  input  logic [15:0]  n_wdone,
  output logic [31:0]  a2x_pub,            // shared beats written and counted (published)
  input  logic [31:0]  x2a_pub,            // XDMA's, answered
  output logic         done,
  output logic         bad
);
  import tb_memch_pkg::*;
  logic [7:0] sh [NB * 64];
  bit         unk [NB * 64];
  logic [7:0] ssh [NSP * 32];              // the split region's bytes 0-31
  bit         sunk [NSP * 32];
  // one command; kind 0: the window (a read's m: the bytes to check), 1: a shared write, 2: a
  // shared read, 3: a split write (si: the split beat), 4: a split read
  typedef struct { bit we; int b; logic [511:0] d; logic [63:0] m; int kind; int si; } op_t;
  typedef struct { logic [511:0] d; logic [63:0] m; } wd_t;
  typedef struct { int b; logic [63:0] m; int kind; int si; bit pub; } pw_t;   // a write not yet counted
  op_t cq [$], rq [$];
  wd_t dq [$];
  pw_t pw [$];
  int ntx = 400, issued = 0, nerr = 0;
  int gapw = 20, gapc = 10, mstall = 30, wpct = 50, seq = 0, ppct = 25, raw = 20, psh = 10, psp = 10;
  int rbuf = 96;
  int rres = 0, occ = 0, nextb = 0, lw_b0 = -1, lw_n = 0;
  int a2x_wr = 0, a2x_p = 0, x2a_rd = 0, x2a_chk = 0;
  longint wacc = 0, wdat = 0, wdone_tot = 0, nrb = 0, nwb = 0, npart = 0, nsh = 0, nsp = 0;
  logic [15:0] wd_prev = '0;
  longint c0 = -1, c1 = 0, cyc = 0;
  assign a2x_pub = 32'(a2x_p);

  initial begin
    void'($value$plusargs("ntx=%d", ntx));
    void'($value$plusargs({NAME, "_ntx=%d"}, ntx));
    void'($value$plusargs("mstall=%d", mstall));
    void'($value$plusargs("wpct=%d", wpct));
    void'($value$plusargs("seq=%d", seq));
    void'($value$plusargs("gapw=%d", gapw));
    void'($value$plusargs("gapc=%d", gapc));
    void'($value$plusargs("ppct=%d", ppct));
    void'($value$plusargs("raw=%d", raw));
    void'($value$plusargs("psh=%d", psh));
    void'($value$plusargs("psp=%d", psp));
    void'($value$plusargs("rbuf=%d", rbuf));
    for (int i = 0; i < NB * 64; i++) begin sh[i] = 8'h00; unk[i] = 1'b0; end
    for (int i = 0; i < NSP * 32; i++) begin ssh[i] = 8'h00; sunk[i] = 1'b0; end
    n_cvalid = 1'b0; n_wvalid = 1'b0;
  end

  // a partial beat's byte enables
  function automatic logic [63:0] pmask();
    logic [63:0] m;
    int a, b;
    case ($urandom % 6)
      0: m = 64'h1 << ($urandom % 64);
      1: m = ~(64'h1 << ($urandom % 64));
      2: begin
        a = $urandom % 64; b = $urandom % 64;
        if (a > b) begin int t; t = a; a = b; b = t; end
        m = '0;
        for (int k = a; k <= b; k++) m[k] = 1'b1;
        if (m == '1) m[63] = 1'b0;
      end
      3: m = '0;
      default: begin m = {$urandom, $urandom}; if (m == '1) m[7] = 1'b0; end
    endcase
    return m;
  endfunction

  always @(posedge clk) begin
    cyc++;
    if (rst) begin
      // writes not counted may or may not land: their bytes become unknown, shared beats not
      // published are written again
      foreach (pw[j]) begin
        if (pw[j].kind == 0) for (int k = 0; k < 64; k++) if (pw[j].m[k]) unk[64 * pw[j].b + k] = 1'b1;
        if (pw[j].kind == 3) for (int k = 0; k < 32; k++) if (pw[j].m[k]) sunk[32 * pw[j].si + k] = 1'b1;
      end
      a2x_wr = a2x_p; x2a_rd = x2a_chk;
      cq.delete(); dq.delete(); rq.delete(); pw.delete();
      rres = 0; occ = 0; wacc = 0; wdat = 0; wdone_tot = 0; wd_prev = '0;
      n_cvalid <= 1'b0; n_wvalid <= 1'b0;
    end else begin
      // ---- n_wdone: never ahead of the writes given (command and data), each counted once
      begin
        logic [15:0] d;
        longint lim;
        d = n_wdone - wd_prev;
        lim = ((wacc < wdat) ? wacc : wdat) - wdone_tot;
        if (longint'(d) > lim) begin
          if (nerr < 20)
            $display("ERROR %s: n_wdone %0d -> %0d with %0d writes given and not yet counted", NAME,
                     wd_prev, n_wdone, lim);
          nerr++;
        end else if (d != 0) begin
          for (int j = 0; j < int'(d); j++) begin
            if (pw[0].pub) a2x_p = pw[0].si + 1;
            void'(pw.pop_front());
          end
          wdone_tot += longint'(d);
          nwb += longint'(d);
          c1 = cyc;
        end
        wd_prev = n_wdone;
      end
      // ---- read data, in command order, into the room kept for it
      if (n_rvalid) begin
        if (rq.size() == 0) begin $display("ERROR %s: read data without a read", NAME); nerr++; end
        else begin
          for (int k = 0; k < 64; k++)
            if (rq[0].m[k] && n_rdata[8 * k +: 8] !== rq[0].d[8 * k +: 8]) begin
              if (nerr < 20)
                $display("ERROR %s: %s beat %0d byte %0d: %h, expected %h", NAME,
                         rq[0].kind == 2 ? "shared" : rq[0].kind == 4 ? "split" : "window", rq[0].b, k,
                         n_rdata[8 * k +: 8],
                         rq[0].d[8 * k +: 8]);
              nerr++;
            end
          if (rq[0].kind == 2) begin x2a_chk++; nsh++; end
          if (rq[0].kind == 4) nsp++;
          void'(rq.pop_front());
          rres--; occ++; nrb++; c1 = cyc;
        end
      end
      if (occ > 0 && ($urandom % 100) >= mstall) occ--;
      // ---- write data, in write order, independent of the commands
      if (n_wvalid && n_wready) begin void'(dq.pop_front()); wdat++; end
      if (!n_wvalid || n_wready) begin
        if (dq.size() != 0 && ($urandom % 100) >= gapw) begin
          n_wvalid <= 1'b1; n_wdata <= dq[0].d; n_wmask <= dq[0].m;
        end else n_wvalid <= 1'b0;
      end
      // ---- commands; a read only with room for its data
      if (n_cvalid && n_cready) begin
        if (cq[0].we) wacc++; else rq.push_back(cq[0]);
        void'(cq.pop_front());
        if (c0 < 0) c0 = cyc;
      end
      if (!n_cvalid || n_cready) begin
        if (cq.size() != 0 && (seq != 0 || ($urandom % 100) >= gapc) && (cq[0].we || rres + occ < rbuf)) begin
          n_cvalid <= 1'b1; n_cwe <= cq[0].we; n_caddr <= 25'(cq[0].b);
          if (!cq[0].we) rres++;
        end else n_cvalid <= 1'b0;
      end
      // ---- the program: a run of beats, or a shared-beat access
      if (issued < ntx && cq.size() < 8) begin
        if (seq == 0 && ($urandom % 100) < psh && (a2x_wr < NS || x2a_rd < int'(x2a_pub))) begin
          if (x2a_rd < int'(x2a_pub) && (a2x_wr >= NS || $urandom % 2 == 0)) begin
            cq.push_back('{1'b0, X2A + x2a_rd, pat(CH, 1, x2a_rd), '1, 2, x2a_rd});
            x2a_rd++;
          end else begin
            logic [511:0] d;
            logic [63:0] m;
            // one whole write, or two partial ones with complementary masks (the second publishes)
            d = pat(CH, 0, a2x_wr);
            m = ($urandom % 2 == 0) ? pmask() : '1;
            if (m != '1) begin
              cq.push_back('{1'b1, A2X + a2x_wr, d, m, 1, a2x_wr});
              dq.push_back('{d, m});
              pw.push_back('{A2X + a2x_wr, m, 1, a2x_wr, 1'b0});
              m = ~m;
            end
            cq.push_back('{1'b1, A2X + a2x_wr, d, m, 1, a2x_wr});
            dq.push_back('{d, m});
            pw.push_back('{A2X + a2x_wr, m, 1, a2x_wr, 1'b1});
            a2x_wr++;
          end
        end else if (seq == 0 && ($urandom % 100) < psp) begin
          // the split region: write some of bytes 0-31 of a beat (always partial), or read them back
          op_t o;
          int i;
          i = $urandom % NSP;
          o.b = SPL + i; o.si = i;
          o.we = ($urandom % 2) == 0;
          o.kind = o.we ? 3 : 4;
          o.m = '0;
          for (int k = 0; k < 32; k++) begin
            if (o.we) begin
              o.d[8 * k +: 8] = 8'($urandom);
              o.m[k] = ($urandom % 4) != 0;
              if (o.m[k]) begin ssh[32 * i + k] = o.d[8 * k +: 8]; sunk[32 * i + k] = 1'b0; end
            end else begin
              o.d[8 * k +: 8] = ssh[32 * i + k];
              o.m[k] = !sunk[32 * i + k];
            end
          end
          if (o.we) begin
            dq.push_back('{o.d, o.m});
            pw.push_back('{o.b, o.m, 3, i, 1'b0});
          end
          cq.push_back(o);
        end else begin
          int n, b0;
          bit wr_, rw;
          rw  = seq == 0 && lw_b0 >= 0 && ($urandom % 100) < raw;
          wr_ = !rw && ($urandom % 100) < wpct;
          n   = seq != 0 ? LMAX : rw ? lw_n : 1 + $urandom % LMAX;
          b0  = seq != 0 ? nextb : rw ? lw_b0 : $urandom % NB;
          if (b0 + n > NB) n = NB - b0;
          nextb = (b0 + n) % NB;
          if (wr_) begin lw_b0 = b0; lw_n = n; end
          for (int j = 0; j < n; j++) begin
            op_t o;
            int b;
            b = b0 + j;
            o.we = wr_; o.b = b; o.kind = 0; o.si = 0;
            if (wr_) begin
              o.m = (($urandom % 100) < ppct) ? pmask() : '1;
              for (int k = 0; k < 64; k++) begin
                o.d[8 * k +: 8] = 8'($urandom);
                if (o.m[k]) begin sh[64 * b + k] = o.d[8 * k +: 8]; unk[64 * b + k] = 1'b0; end
              end
              if (o.m != '1) npart++;
              dq.push_back('{o.d, o.m});
              pw.push_back('{b, o.m, 0, 0, 1'b0});
            end else begin
              for (int k = 0; k < 64; k++) begin
                o.d[8 * k +: 8] = sh[64 * b + k];
                o.m[k] = !unk[64 * b + k];
              end
            end
            cq.push_back(o);
          end
        end
        issued++;
      end
    end
  end
  assign done = (issued >= ntx && cq.size() == 0 && dq.size() == 0 && rq.size() == 0 && !n_cvalid &&
                 !n_wvalid && wdone_tot == wacc && wacc == wdat) || nerr != 0;
  assign bad  = nerr != 0;

  final begin
    $display("%s: %0d runs, %0d read beats, %0d write beats counted (%0d partial), %0d shared read, %0d shared published, %0d split read, %0d errors, %.3f beats per cycle",
             NAME, issued, nrb, nwb, npart, nsh, a2x_p, nsp, nerr, real'(nrb + nwb) / real'(c1 - c0 + 1));
  end
endmodule

// A random AXI4 master over its windows (XDMA; see tb_memch): INCR bursts of BYTES-byte beats
// (BYTES = 16: the shared and split accesses are 16-byte lanes).
module tb_memch_axi #(
  parameter int BYTES = 16,
  parameter int IDW = 4,
  parameter logic [31:0] WIN0 = 32'h0,     // its windows, one per channel
  parameter logic [31:0] WIN1 = 32'h0,
  parameter logic [31:0] SPAN = 32'h8000,
  parameter int LMAX = 64,
  parameter int OUTS = 8,                  // bursts in flight per direction
  parameter int A2X = 0,                   // shared regions (beats): the ones it reads, the ones it writes
  parameter int X2A = 0,
  parameter int NS = 256,
  parameter int SPL = 0,                   // split region: it owns bytes 32-63 of each beat
  parameter int NSP = 64,
  parameter string NAME = "xdma"
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
  input  logic [1:0][31:0]     a2x_pub,
  output logic [1:0][31:0]     x2a_pub,
  output logic                 done,
  output logic                 bad
);
  import tb_memch_pkg::*;
  localparam int NB = 2 * SPAN;             // shadow bytes
  logic [7:0] sh [NB];
  bit         unk [NB];
  int         wp [NB / 64], rp [NB / 64];   // writes / reads in flight per 64-byte beat
  logic [7:0] ssh [2 * NSP * 32];           // the split regions' bytes 32-63 (channel, beat)
  bit         sunk [2 * NSP * 32];
  int         swp [2 * NSP], srp [2 * NSP];
  typedef struct { logic [BYTES*8-1:0] d; logic [BYTES-1:0] s; bit last; } w_t;
  // one R beat: its window byte, or (shd) the expected data and the bytes to check
  typedef struct { int idx; logic [IDW-1:0] id; bit last; bit shd; logic [BYTES*8-1:0] exp; logic [BYTES-1:0] chk; } r_t;
  // a burst's 64-byte beats in the window; sc: a shared beat's channel; sp: a split beat
  typedef struct { int b0; int nb; logic [IDW-1:0] id; int sc; int sp; } x_t;
  w_t wq [$];
  r_t rq [$];
  x_t bq [$], rdq [$];
  int ntx = 400, issued = 0, nerr = 0;
  longint nrb = 0, nwb = 0, nsh = 0, nsp = 0;
  longint nwl = 0, nbr = 0;                 // bursts whose last W beat went, B responses
  int gapw = 20, mstall = 30, wpct = 50, seq = 0, xfull = 70, psh = 10, psp = 10;
  int outs = OUTS;                          // +outs: bursts in flight per direction
  // +awdly=N: a write burst's AW goes out 0..N-1 cycles after its W beats are queued (in burst
  // order), so W runs ahead of AW by up to +outs bursts; +wlen=N: N-beat bursts (e.g. 8: 128 B)
  int awdly = 0, wlen = 0;
  typedef struct { logic [31:0] a; logic [7:0] len; logic [IDW-1:0] id; longint t; } aw_t;
  aw_t awq [$];
  int a2x_rd [2], a2x_chk [2], x2a_wr [2];
  logic [31:0] nexto = 0;                   // +seq: the next burst's offset
  longint c0 = -1, c1 = 0, cyc = 0;         // first and last data cycles

  // without +awdly the AW goes out with its burst (the channel must be free); with it, through awq
  logic aw_free;
  assign aw_free = awdly == 0 ? !(awvalid && !awready) : awq.size() < 64;
  task automatic aw_issue(input logic [31:0] a, input logic [7:0] len, input logic [IDW-1:0] id);
    if (awdly == 0) begin awvalid <= 1'b1; awaddr <= a; awlen <= len; awid <= id; end
    else awq.push_back('{a, len, id, cyc + longint'($urandom % awdly)});
  endtask

  function automatic int idx_of(input logic [31:0] a);
    if (a[31] != WIN0[31]) return int'(SPAN + (a - WIN1));
    return int'(a - WIN0);
  endfunction

  initial begin
    void'($value$plusargs("ntx=%d", ntx));
    void'($value$plusargs({NAME, "_ntx=%d"}, ntx));
    void'($value$plusargs("mstall=%d", mstall));
    void'($value$plusargs("wpct=%d", wpct));
    void'($value$plusargs("seq=%d", seq));
    void'($value$plusargs("gapw=%d", gapw));
    void'($value$plusargs("xfull=%d", xfull));
    void'($value$plusargs("psh=%d", psh));
    void'($value$plusargs("psp=%d", psp));
    void'($value$plusargs("outs=%d", outs));
    void'($value$plusargs("awdly=%d", awdly));
    void'($value$plusargs("wlen=%d", wlen));
    for (int i = 0; i < NB; i++) begin sh[i] = 8'h00; unk[i] = 1'b0; end
    for (int i = 0; i < NB / 64; i++) begin wp[i] = 0; rp[i] = 0; end
    for (int i = 0; i < 2 * NSP * 32; i++) begin ssh[i] = 8'h00; sunk[i] = 1'b0; end
    for (int i = 0; i < 2 * NSP; i++) begin swp[i] = 0; srp[i] = 0; end
    for (int c = 0; c < 2; c++) begin a2x_rd[c] = 0; a2x_chk[c] = 0; x2a_wr[c] = 0; end
    x2a_pub = '0;
  end

  always @(posedge clk) begin
    cyc++;
    if ((rvalid && rready) || (wvalid && wready)) begin if (c0 < 0) c0 = cyc; c1 = cyc; end
    if (rst) begin
      // writes in flight may or may not land: their beats become unknown; shared beats not yet
      // answered are written again, shared reads in flight read again
      foreach (bq[j]) begin
        for (int b = bq[j].b0; b < bq[j].b0 + bq[j].nb; b++)
          for (int k = 0; k < 64; k++) unk[64 * b + k] = 1'b1;
        if (bq[j].sp >= 0) for (int k = 0; k < 32; k++) sunk[32 * bq[j].sp + k] = 1'b1;
      end
      for (int i = 0; i < NB / 64; i++) begin wp[i] = 0; rp[i] = 0; end
      for (int i = 0; i < 2 * NSP; i++) begin swp[i] = 0; srp[i] = 0; end
      for (int c = 0; c < 2; c++) begin x2a_wr[c] = int'(x2a_pub[c]); a2x_rd[c] = a2x_chk[c]; end
      wq.delete(); rq.delete(); bq.delete(); rdq.delete(); awq.delete();
      nwl = 0; nbr = 0;
      awvalid <= 1'b0; wvalid <= 1'b0; arvalid <= 1'b0; bready <= 1'b0; rready <= 1'b0;
    end else begin
      bit whs;
      // ---- responses
      if (bvalid && bready) begin
        if (bq.size() == 0) begin $display("ERROR %s: B without a write", NAME); nerr++; end
        else if (nbr >= nwl) begin
          $display("ERROR %s: B id %h before its burst's last W beat", NAME, bid); nerr++;
        end else begin
          if (bid != bq[0].id || bresp != 0) begin
            $display("ERROR %s: B id %h resp %h, expected id %h", NAME, bid, bresp, bq[0].id); nerr++;
          end
          for (int b = bq[0].b0; b < bq[0].b0 + bq[0].nb; b++) wp[b]--;
          if (bq[0].sc >= 0) x2a_pub[bq[0].sc] <= x2a_pub[bq[0].sc] + 1;
          if (bq[0].sp >= 0) swp[bq[0].sp]--;
          void'(bq.pop_front());
          nbr++;
        end
      end
      if (rvalid && rready) begin
        if (rq.size() == 0) begin $display("ERROR %s: R without a read", NAME); nerr++; end
        else begin
          for (int k = 0; k < BYTES; k++)
            if (rq[0].shd ? (rq[0].chk[k] && rdata[8 * k +: 8] !== rq[0].exp[8 * k +: 8])
                          : (!unk[rq[0].idx + k] && rdata[8 * k +: 8] !== sh[rq[0].idx + k])) begin
              if (nerr < 20)
                $display("ERROR %s: %s read byte %0d: %h, expected %h", NAME,
                         rq[0].shd ? "shared/split" : "window", rq[0].shd ? k : rq[0].idx + k,
                         rdata[8 * k +: 8], rq[0].shd ? rq[0].exp[8 * k +: 8] : sh[rq[0].idx + k]);
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
            if (rdq[0].sc >= 0) begin a2x_chk[rdq[0].sc]++; nsh++; end
            if (rdq[0].sp >= 0) begin srp[rdq[0].sp]--; nsp++; end
            void'(rdq.pop_front());
          end
          void'(rq.pop_front());
        end
      end
      bready <= ($urandom % 100) >= mstall;
      rready <= ($urandom % 100) >= mstall;
      // ---- W
      whs = wvalid && wready;
      if (whs) begin void'(wq.pop_front()); nwb++; if (wlast) nwl++; end
      if (!wvalid || whs) begin
        if (wq.size() != 0 && ($urandom % 100) >= gapw) begin
          wvalid <= 1'b1; wdata <= wq[0].d; wstrb <= wq[0].s; wlast <= wq[0].last;
        end else wvalid <= 1'b0;
      end
      // ---- AW / AR: a new burst when the channel is free
      if (awvalid && awready) awvalid <= 1'b0;
      if (arvalid && arready) arvalid <= 1'b0;
      if (issued < ntx && ($urandom % 100) < 50) begin
        int c;
        logic [IDW-1:0] id;
        c = $urandom % 2;
        id = IDW'($urandom);
        if (seq == 0 && ($urandom % 100) < psh) begin
          // a shared beat: read one the accelerator has published, or write the next one
          bit rd_;
          logic [511:0] p;
          rd_ = a2x_rd[c] < int'(a2x_pub[c]) && (x2a_wr[c] >= NS || $urandom % 2 == 0);
          if (rd_ && !(arvalid && !arready) && rdq.size() < outs) begin
            p = pat(c, 0, a2x_rd[c]);
            for (int j = 0; j < 4; j++) rq.push_back('{0, id, j == 3, 1'b1, p[128 * j +: 128], '1});
            rdq.push_back('{0, 0, id, c, -1});
            arvalid <= 1'b1; araddr <= (32'(c) << 31) | (32'(A2X + a2x_rd[c]) << 6); arlen <= 8'd3; arid <= id;
            a2x_rd[c]++;
            issued++;
          end else if (!rd_ && x2a_wr[c] < NS && aw_free && bq.size() < outs) begin
            p = pat(c, 1, x2a_wr[c]);
            for (int j = 0; j < 4; j++) wq.push_back('{p[128 * j +: 128], '1, j == 3});
            bq.push_back('{0, 0, id, c, -1});
            aw_issue((32'(c) << 31) | (32'(X2A + x2a_wr[c]) << 6), 8'd3, id);
            x2a_wr[c]++;
            issued++;
          end
        end else if (seq == 0 && ($urandom % 100) < psp) begin
          // the split region: write lanes 2-3 of a beat (a partial 64-byte beat), or read them back
          int s;
          bit wr_;
          logic [31:0] a;
          s = c * NSP + $urandom % NSP;
          wr_ = ($urandom % 2) == 0;
          a = (32'(c) << 31) | (32'(SPL + s % NSP) << 6) | 32'd32;
          if (wr_ && swp[s] == 0 && srp[s] == 0 && aw_free && bq.size() < outs) begin
            for (int j = 0; j < 2; j++) begin
              w_t x;
              for (int k = 0; k < BYTES; k++) begin
                x.d[8 * k +: 8] = 8'($urandom);
                x.s[k] = ($urandom % 4) != 0;
                if (x.s[k]) begin ssh[32 * s + 16 * j + k] = x.d[8 * k +: 8]; sunk[32 * s + 16 * j + k] = 1'b0; end
              end
              x.last = j == 1;
              wq.push_back(x);
            end
            swp[s]++;
            bq.push_back('{0, 0, id, -1, s});
            aw_issue(a, 8'd1, id);
            issued++;
          end else if (!wr_ && swp[s] == 0 && !(arvalid && !arready) && rdq.size() < outs) begin
            for (int j = 0; j < 2; j++) begin
              r_t r;
              r.idx = 0; r.id = id; r.last = j == 1; r.shd = 1'b1;
              for (int k = 0; k < BYTES; k++) begin
                r.exp[8 * k +: 8] = ssh[32 * s + 16 * j + k];
                r.chk[k] = !sunk[32 * s + 16 * j + k];
              end
              rq.push_back(r);
            end
            srp[s]++;
            rdq.push_back('{0, 0, id, -1, s});
            arvalid <= 1'b1; araddr <= a; arlen <= 8'd1; arid <= id;
            issued++;
          end
        end else begin
          bit wr_;
          int w, n, i0, b0, b1;
          logic [31:0] off, a;
          bit ok;
          wr_ = ($urandom % 100) < wpct;
          w = $urandom % 2;
          off = seq != 0 ? nexto : ($urandom % (SPAN / BYTES)) * BYTES;
          a = (w != 0 ? WIN1 : WIN0) + off;
          n = seq != 0 ? LMAX : wlen != 0 ? wlen : 1 + $urandom % LMAX;
          if (n * BYTES > 4096 - (a & 4095)) n = (4096 - (a & 4095)) / BYTES;
          if (n * BYTES > SPAN - off) n = (SPAN - off) / BYTES;
          i0 = idx_of(a);
          b0 = i0 / 64; b1 = (i0 + n * BYTES - 1) / 64;
          ok = wr_ ? (aw_free && bq.size() < outs) : (!(arvalid && !arready) && rdq.size() < outs);
          for (int b = b0; b <= b1 && ok; b++) if (wp[b] != 0 || (wr_ && rp[b] != 0)) ok = 1'b0;
          if (ok) begin
            nexto = (off + n * BYTES) % SPAN;
            issued++;
            if (wr_) begin
              bit full;
              full = ($urandom % 100) < xfull;
              for (int j = 0; j < n; j++) begin
                w_t x;
                for (int k = 0; k < BYTES; k++) begin
                  x.d[8 * k +: 8] = 8'($urandom);
                  x.s[k] = full || ($urandom % 2) != 0;
                  if (x.s[k]) begin sh[i0 + j * BYTES + k] = x.d[8 * k +: 8]; unk[i0 + j * BYTES + k] = 1'b0; end
                end
                x.last = j == n - 1;
                wq.push_back(x);
              end
              for (int b = b0; b <= b1; b++) wp[b]++;
              bq.push_back('{b0, b1 - b0 + 1, id, -1, -1});
              aw_issue(a, 8'(n - 1), id);
            end else begin
              for (int j = 0; j < n; j++) rq.push_back('{i0 + j * BYTES, id, j == n - 1, 1'b0, '0, '0});
              for (int b = b0; b <= b1; b++) rp[b]++;
              rdq.push_back('{b0, b1 - b0 + 1, id, -1, -1});
              arvalid <= 1'b1; araddr <= a; arlen <= 8'(n - 1); arid <= id;
            end
          end
        end
      end
      // ---- the delayed AWs (+awdly), in burst order
      if (awdly != 0 && (!awvalid || awready) && awq.size() != 0 && awq[0].t <= cyc) begin
        awvalid <= 1'b1; awaddr <= awq[0].a; awlen <= awq[0].len; awid <= awq[0].id;
        void'(awq.pop_front());
      end
    end
  end
  assign done = (issued >= ntx && bq.size() == 0 && rdq.size() == 0 && wq.size() == 0 && !awvalid &&
                 awq.size() == 0 &&
                 !arvalid) || nerr != 0;
  assign bad  = nerr != 0;

  final begin
    $display("%s: %0d bursts, %0d read beats, %0d write beats, %0d shared read, %0d shared published, %0d split read, %0d errors, %.3f beats per cycle",
             NAME, issued, nrb, nwb, nsh, x2a_pub[0] + x2a_pub[1], nsp, nerr, real'(nrb + nwb) / real'(c1 - c0 + 1));
  end
endmodule
