// Test of the debug build's DMA monitors (rtl/boards/ypcb-00338/otpu_xmon.sv) on the card's path:
// an XDMA master (128-bit AXI, xclk) through otpu_axi_split2 onto both channels' otpu_mem_ch, each
// in front of a controller model (LDC = 1: LiteDRAM's own controller, otpu_ldc_model; 0:
// otpu_ldn_model), the accelerators idle. otpu_xmon watches the master and channel 0's controller
// ports, as otpu_native_sys wires it, and the bench reads its words at the end through its
// register side as the host does (SNAP, the acknowledgements, the shadows).
//
// The master writes self-describing data (each 16-byte beat: MAGIC, its address, the address
// inverted, the burst's tag): first both windows whole (256-byte bursts), then +ntx random writes
// and +ntx random reads at once (1 to +maxlen beats from any 16-byte offset, so partial 64-byte
// beats go through otpu_mem_ch's read-modify-write), up to +outs bursts of each in flight, with
// valid gaps (+gap) and B / R backpressure (+stall), and checks every read beat itself.
// +inj=K (at event +injn=N) puts in one fault, where the card's could be:
//   1 XDMA: the W data from burst tag N on describes the address 64 bytes below its own (the
//     card's slip)
//   2 otpu_mem_ch: channel 0 port 0's N-th write data beat is the beat before it again
//   3 XDMA: after N W beats, no more W (the card's stuck engine)
//   4 downstream: after N W beats, W ready stays low
//   5 controller: channel 0's N-th read data beat is the beat before it again
//   6 split: the N-th R beat XDMA sees is the beat before it again
//   7 downstream: after N B responses, B valid stays low
// The end: the master done, or nothing moving for 3 x 2^TW cycles. Prints the monitor's words, the
// master's own count of bad read beats and one RESULT line.
module tb_xmon #(
  parameter int LDC = 1,
  parameter int TW = 12
);
  localparam logic [31:0] MAGIC = 32'h584D_4F4E;
  int cp = 500, up = 375, xp = 400;        // half periods: 100 / 133.3 / 125 MHz (B, FAST)
  logic clk = 1'b0, uclk = 1'b0, xclk = 1'b0;
  int inj = 0, injn = 0;
  initial begin
    int seed;
    void'($value$plusargs("cp=%d", cp));
    void'($value$plusargs("up=%d", up));
    void'($value$plusargs("xp=%d", xp));
    void'($value$plusargs("inj=%d", inj));
    void'($value$plusargs("injn=%d", injn));
    if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));
  end
  always #(cp) clk = ~clk;
  always #(up) uclk = ~uclk;
  always #(xp) xclk = ~xclk;

  logic rst = 1'b1, urst = 1'b1, xrst = 1'b1;
  longint ccyc = 0, ucyc = 0, xcyc = 0;
  always @(posedge clk) begin ccyc <= ccyc + 1; if (ccyc == 30) rst <= 1'b0; end
  always @(posedge uclk) begin ucyc <= ucyc + 1; if (ucyc == 20) urst <= 1'b0; end
  always @(posedge xclk) begin xcyc <= xcyc + 1; if (xcyc == 40) xrst <= 1'b0; end

  // ---- the master (d_*: as XDMA drives and sees them, which otpu_xmon watches) and the split (s_*)
  logic d_awv, d_awr, d_wv, d_wr, d_wl, d_bv, d_br, d_arv, d_arr, d_rv, d_rr, d_rl;
  logic [3:0] d_awi, d_bi, d_ari, d_ri;
  logic [31:0] d_awa, d_ara;
  logic [7:0] d_awl, d_arl;
  logic [127:0] d_wd, d_rd, s_rd;
  logic [15:0] d_ws;
  logic [1:0] d_bre, d_rre;
  logic s_wv, s_wr, s_bv, s_br;
  logic xdone;
  int xerrs, xrchk;
  longint nwb, nb, nrb;

  tb_xmon_dma #(.MAGIC(MAGIC)) u_dma (
    .clk(xclk), .rst(xrst), .inj, .injn,
    .awvalid(d_awv), .awready(d_awr), .awid(d_awi), .awaddr(d_awa), .awlen(d_awl),
    .wvalid(d_wv), .wready(d_wr), .wdata(d_wd), .wstrb(d_ws), .wlast(d_wl),
    .bvalid(d_bv), .bready(d_br), .bid(d_bi),
    .arvalid(d_arv), .arready(d_arr), .arid(d_ari), .araddr(d_ara), .arlen(d_arl),
    .rvalid(d_rv), .rready(d_rr), .rid(d_ri), .rdata(d_rd), .rlast(d_rl),
    .done(xdone), .errs(xerrs), .rchk(xrchk), .nwb, .nb, .nrb);

  // faults between the master and the split
  wire w_hold = inj == 4 && nwb >= longint'(injn);
  wire b_hold = inj == 7 && nb >= longint'(injn);
  assign s_wv = d_wv && !w_hold;
  assign d_wr = s_wr && !w_hold;
  assign d_bv = s_bv && !b_hold;
  assign s_br = d_br && !b_hold;
  logic [127:0] r_prev = '0;
  always @(posedge xclk) if (d_rv && d_rr) r_prev <= d_rd;
  assign d_rd = (inj == 6 && nrb == longint'(injn)) ? r_prev : s_rd;

  logic [1:0] cawv, cawr, cwvx, cwrx, cbv, cbr, carv, carr, crvx, crrx, crl;
  logic [1:0][3:0] cbi, cri;
  logic [1:0][1:0] cbre, crre;
  logic [1:0][127:0] crdx;
  otpu_axi_split2 #(.IDW(4), .DW(128)) u_split (
    .clk(xclk), .rst(xrst),
    .s_awvalid(d_awv), .s_awready(d_awr), .s_awid(d_awi), .s_awaddr(d_awa), .s_awlen(d_awl),
    .s_wvalid(s_wv), .s_wready(s_wr), .s_wdata(d_wd), .s_wstrb(d_ws), .s_wlast(d_wl),
    .s_bvalid(s_bv), .s_bready(s_br), .s_bid(d_bi), .s_bresp(d_bre),
    .s_arvalid(d_arv), .s_arready(d_arr), .s_arid(d_ari), .s_araddr(d_ara), .s_arlen(d_arl),
    .s_rvalid(d_rv), .s_rready(d_rr), .s_rid(d_ri), .s_rdata(s_rd), .s_rresp(d_rre), .s_rlast(d_rl),
    .m_awvalid(cawv), .m_awready(cawr), .m_wvalid(cwvx), .m_wready(cwrx),
    .m_bvalid(cbv), .m_bready(cbr), .m_bid(cbi), .m_bresp(cbre),
    .m_arvalid(carv), .m_arready(carr),
    .m_rvalid(crvx), .m_rready(crrx), .m_rid(cri), .m_rdata(crdx), .m_rresp(crre), .m_rlast(crl));

  // ---- the bridges and the controllers; c_*: as the controllers see them (otpu_xmon: channel 0)
  logic [1:0][1:0] ccv, ccr, ccwe, cwv, cwr, crv;
  logic [1:0][1:0][24:0] cca;
  logic [1:0][1:0][511:0] cwd, crd, mwd, mrd;
  logic [1:0][1:0][63:0] cwe;
  logic dump = 1'b0;
  for (genvar c = 0; c < 2; c++) begin : g_ch
    logic [24:0] nca;
    logic [511:0] nwd, nrd;
    logic [63:0] nwm;
    logic [15:0] nwdone;
    logic ncr, nwr, nrv, nerr;
    assign nca = '0; assign nwd = '0; assign nwm = '0;
    otpu_mem_ch #(.XIDW(4)) u_ch (
      .clk, .rst,
      .n_cvalid(1'b0), .n_cready(ncr), .n_cwe(1'b0), .n_caddr(nca),
      .n_wvalid(1'b0), .n_wready(nwr), .n_wdata(nwd), .n_wmask(nwm),
      .n_rvalid(nrv), .n_rdata(nrd), .n_wdone(nwdone), .n_err(nerr),
      .xclk, .xrst,
      .x_awvalid(cawv[c]), .x_awready(cawr[c]), .x_awid(d_awi), .x_awaddr(d_awa), .x_awlen(d_awl),
      .x_wvalid(cwvx[c]), .x_wready(cwrx[c]), .x_wdata(d_wd), .x_wstrb(d_ws), .x_wlast(d_wl),
      .x_bvalid(cbv[c]), .x_bready(cbr[c]), .x_bid(cbi[c]), .x_bresp(cbre[c]),
      .x_arvalid(carv[c]), .x_arready(carr[c]), .x_arid(d_ari), .x_araddr(d_ara), .x_arlen(d_arl),
      .x_rvalid(crvx[c]), .x_rready(crrx[c]), .x_rid(cri[c]), .x_rdata(crdx[c]), .x_rresp(crre[c]),
      .x_rlast(crl[c]),
      .uclk, .urst,
      .c_cmd_valid(ccv[c]), .c_cmd_ready(ccr[c]), .c_cmd_we(ccwe[c]), .c_cmd_addr(cca[c]),
      .c_wdata_valid(cwv[c]), .c_wdata_ready(cwr[c]), .c_wdata_data(mwd[c]),
      .c_wdata_we(cwe[c]), .c_rdata_valid(crv[c]), .c_rdata_data(crd[c]));
    always @(posedge uclk) if (!urst && nerr) begin
      $display("FAIL channel %0d: otpu_mem_ch n_err", c);
      $finish;
    end
    if (LDC) begin : g_ldc
      otpu_ldc_model #(.CH(c)) u_mem (
        .clk(uclk), .rst(urst),
        .c_cmd_valid(ccv[c]), .c_cmd_ready(ccr[c]), .c_cmd_we(ccwe[c]), .c_cmd_addr(cca[c]),
        .c_wdata_valid(cwv[c]), .c_wdata_ready(cwr[c]), .c_wdata_data(cwd[c]),
        .c_wdata_we(cwe[c]), .c_rdata_valid(crv[c]), .c_rdata_data(mrd[c]), .dump);
    end else begin : g_ldn
      otpu_ldn_model #(.BEATS(1 << 15), .CH(c)) u_mem (
        .clk(uclk), .rst(urst),
        .c_cmd_valid(ccv[c]), .c_cmd_ready(ccr[c]), .c_cmd_we(ccwe[c]), .c_cmd_addr(cca[c]),
        .c_wdata_valid(cwv[c]), .c_wdata_ready(cwr[c]), .c_wdata_data(cwd[c]),
        .c_wdata_we(cwe[c]), .c_rdata_valid(crv[c]), .c_rdata_data(mrd[c]), .dump);
    end
  end
  // faults between channel 0's bridge and its controller
  longint nw00 = 0, nr0 = 0;
  logic [511:0] w00_prev = '0, r0_prev = '0;
  always @(posedge uclk) begin
    if (cwv[0][0] && cwr[0][0]) begin nw00 <= nw00 + 1; w00_prev <= cwd[0][0]; end
    if (|crv[0]) begin nr0 <= nr0 + 1; r0_prev <= crd[0][0]; end
  end
  assign cwd[0][0] = (inj == 2 && nw00 == longint'(injn)) ? w00_prev : mwd[0][0];
  assign cwd[0][1] = mwd[0][1];
  assign cwd[1] = mwd[1];
  assign crd[0][0] = (inj == 5 && nr0 == longint'(injn)) ? r0_prev : mrd[0][0];
  assign crd[0][1] = (inj == 5 && nr0 == longint'(injn)) ? r0_prev : mrd[0][1];
  assign crd[1] = mrd[1];

  // ---- the monitors, and the host's side of their registers
  logic snap = 1'b0, clr = 1'b0;
  logic [31:0] word [32];
  otpu_xmon #(.TW(TW), .MAGIC(MAGIC)) u_xmon (
    .clk, .snap, .clr, .word,
    .xclk, .xrst,
    .x_awvalid(d_awv), .x_awready(d_awr), .x_awid(d_awi), .x_awaddr(d_awa), .x_awlen(d_awl),
    .x_wvalid(d_wv), .x_wready(d_wr), .x_wdata(d_wd), .x_wlast(d_wl),
    .x_bvalid(d_bv), .x_bready(d_br), .x_bid(d_bi),
    .x_arvalid(d_arv), .x_arready(d_arr), .x_arid(d_ari), .x_araddr(d_ara), .x_arlen(d_arl),
    .x_rvalid(d_rv), .x_rready(d_rr), .x_rid(d_ri), .x_rdata(d_rd), .x_rlast(d_rl),
    .uclk, .urst,
    .c_cmd_valid(ccv[0]), .c_cmd_ready(ccr[0]), .c_cmd_we(ccwe[0]), .c_cmd_addr(cca[0]),
    .c_wdata_valid(cwv[0]), .c_wdata_ready(cwr[0]), .c_wdata_data(cwd[0]),
    .c_rdata_valid(crv[0]), .c_rdata_data(crd[0]));

  // the end: the master done, or nothing moving on its master for 3 x 2^TW xclk cycles
  longint idle = 0, tmax = 4000000;
  logic fin = 1'b0;
  initial void'($value$plusargs("tmax=%d", tmax));
  always @(posedge xclk) begin
    if ((d_awv && d_awr) || (d_wv && d_wr) || (d_bv && d_br) || (d_arv && d_arr) || (d_rv && d_rr))
      idle <= 0;
    else idle <= idle + 1;
    if (!xrst && (xdone || idle > 3 * (longint'(1) << TW) || xcyc > tmax)) fin <= 1'b1;
  end
  localparam string NAMES [32] = '{"FLAGS", "SNAPS", "X_AW", "X_W", "X_B", "X_AR", "X_R", "X_WCHK",
    "X_RCHK", "X_WBAD", "X_RBAD", "X_WEXP", "X_WGOT", "X_WTAG", "X_REXP", "X_RGOT", "X_STALL",
    "X_STALLN", "X_LIVE", "X_LIVEN", "N_WCMD", "N_WDAT", "N_RCMD", "N_RDAT", "N_WCHK", "N_RCHK",
    "N_WBAD", "N_RBAD", "N_WEXP", "N_WGOT", "N_REXP", "N_RGOT"};
  initial begin
    wait (fin);
    repeat (20) @(posedge clk);
    snap <= 1'b1;
    @(posedge clk);
    snap <= 1'b0;
    repeat (200) begin
      @(posedge clk);
      if (word[1][15:8] == word[1][7:0] && word[1][23:16] == word[1][7:0]) break;
    end
    for (int k = 0; k < 32; k++) $display("XMON %-8s %08h %0d", NAMES[k], word[k], word[k]);
    $display("RESULT inj=%0d done=%0d flags=%04h xerrs=%0d xrchk=%0d nwb=%0d nb=%0d nrb=%0d snaps=%06h",
             inj, xdone, word[0][15:0], xerrs, xrchk, nwb, nb, nrb, word[1][23:0]);
    // CLEAR, then SNAP again: everything zero but the live words, if the master has stopped
    clr <= 1'b1;
    @(posedge clk);
    clr <= 1'b0;
    repeat (20) @(posedge clk);
    snap <= 1'b1;
    @(posedge clk);
    snap <= 1'b0;
    repeat (200) begin
      @(posedge clk);
      if (word[1][15:8] == word[1][7:0] && word[1][23:16] == word[1][7:0]) break;
    end
    begin
      int nz;
      nz = word[0][15:0] != 0 || word[1][7:0] != 8'd2 || word[1][15:8] != 8'd2 || word[1][23:16] != 8'd2;
      for (int k = 2; k < 32; k++) if (k != 18 && k != 19 && word[k] != 0) nz++;
      $display("CLEAR %s", nz == 0 ? "ok" : "BAD");
    end
    $finish;
  end
endmodule

// The XDMA master: self-describing writes, reads checked (see tb_xmon).
module tb_xmon_dma #(
  parameter logic [31:0] MAGIC = 32'h0,
  parameter logic [31:0] BASE = 32'h0010_0000,  // each channel's window: BASE .. BASE + SPAN
  parameter int SPAN = 'h1_0000
) (
  input  logic         clk,
  input  logic         rst,
  input  int           inj,
  input  int           injn,
  output logic         awvalid,
  input  logic         awready,
  output logic [3:0]   awid,
  output logic [31:0]  awaddr,
  output logic [7:0]   awlen,
  output logic         wvalid,
  input  logic         wready,
  output logic [127:0] wdata,
  output logic [15:0]  wstrb,
  output logic         wlast,
  input  logic         bvalid,
  output logic         bready,
  input  logic [3:0]   bid,
  output logic         arvalid,
  input  logic         arready,
  output logic [3:0]   arid,
  output logic [31:0]  araddr,
  output logic [7:0]   arlen,
  input  logic         rvalid,
  output logic         rready,
  input  logic [3:0]   rid,
  input  logic [127:0] rdata,
  input  logic         rlast,
  output logic         done,
  output int           errs,
  output int           rchk,
  output longint       nwb,
  output longint       nb,
  output longint       nrb
);
  typedef struct { logic [31:0] a; int len; int tag; } burst_t;
  int ntx = 2000, outs = 8, gap = 20, stall = 30, maxlen = 16;
  initial begin
    void'($value$plusargs("ntx=%d", ntx));
    void'($value$plusargs("outs=%d", outs));
    void'($value$plusargs("gap=%d", gap));
    void'($value$plusargs("stall=%d", stall));
    void'($value$plusargs("maxlen=%d", maxlen));
  end

  function automatic logic [127:0] beat(input logic [31:0] a, input int tag);
    return {32'(tag), ~a, a, MAGIC};
  endfunction
  function automatic burst_t rnd(input int tag);
    burst_t b;
    int off;
    b.len = 1 + int'($urandom % 32'(maxlen));
    off = int'($urandom % 32'(SPAN / 16 - b.len)) * 16;
    if (off % 4096 + b.len * 16 > 4096) off = off - (off % 4096 + b.len * 16 - 4096);
    b.a = ($urandom % 2 != 0 ? 32'h8000_0000 : 32'h0) | (BASE + 32'(off));
    b.tag = tag;
    return b;
  endfunction

  burst_t wq [$], rq [$], aw_b, ar_b;
  int phase = 0, npre = 0, wk = 0, rk = 0, b_out = 0, r_out = 0, nw = 0, nr = 0, tag = 0;
  localparam int NPRE = 2 * SPAN / 256;
  always @(posedge clk) begin
    if (rst) begin
      awvalid <= 1'b0; wvalid <= 1'b0; arvalid <= 1'b0; bready <= 1'b0; rready <= 1'b0;
      wq.delete(); rq.delete();
      wk = 0; rk = 0; b_out = 0; r_out = 0;
      done <= 1'b0;
    end else begin
      // ---- AW: a burst counts toward outs from its AW until its B
      if (!awvalid || awready) begin
        awvalid <= 1'b0;
        if (b_out < outs && $urandom % 100 >= 32'(gap)) begin
          if (phase == 0 && npre < NPRE) begin
            aw_b.len = 16;
            aw_b.a = (npre % 2 != 0 ? 32'h8000_0000 : 32'h0) | (BASE + 32'(npre / 2 * 256));
            aw_b.tag = tag;
            npre++;
          end else if (phase == 1 && nw < ntx) begin
            aw_b = rnd(tag);
            nw++;
          end else aw_b.len = 0;
          if (aw_b.len != 0) begin
            awvalid <= 1'b1;
            awaddr <= aw_b.a;
            awlen <= 8'(aw_b.len - 1);
            awid <= 4'(tag);
            wq.push_back(aw_b);
            b_out++;
            tag++;
          end
        end
      end
      // ---- W: the bursts in AW order (a W beat may come before its AW is taken)
      if (wvalid && wready) begin
        nwb <= nwb + 1;
        wk++;
        if (wk == wq[0].len) begin
          void'(wq.pop_front());
          wk = 0;
        end
      end
      if (!wvalid || wready) begin
        wvalid <= 1'b0;
        if (wq.size() != 0 && !(inj == 3 && nwb + longint'(wvalid && wready) >= longint'(injn))
            && $urandom % 100 >= 32'(gap)) begin
          logic [31:0] a;
          a = wq[0].a + 32'(16 * wk);
          wvalid <= 1'b1;
          wdata <= beat((inj == 1 && wq[0].tag >= injn) ? a - 32'd64 : a, wq[0].tag);
          wstrb <= '1;
          wlast <= wk == wq[0].len - 1;
        end
      end
      // ---- B
      if (bvalid && bready) begin
        b_out--;
        nb <= nb + 1;
      end
      bready <= $urandom % 100 >= 32'(stall);
      // ---- AR (after the windows are written)
      if (phase == 0 && npre == NPRE && b_out == 0) phase = 1;
      if (!arvalid || arready) begin
        arvalid <= 1'b0;
        if (phase == 1 && nr < ntx && r_out < outs && $urandom % 100 >= 32'(gap)) begin
          ar_b = rnd(0);
          arvalid <= 1'b1;
          araddr <= ar_b.a;
          arlen <= 8'(ar_b.len - 1);
          arid <= 4'($urandom);
          rq.push_back(ar_b);
          r_out++;
          nr++;
        end
      end
      // ---- R: each beat its own address
      if (rvalid && rready) begin
        logic [31:0] a;
        nrb <= nrb + 1;
        a = rq[0].a + 32'(16 * rk);
        if (rdata == beat(a, int'(rdata[127:96]))) rchk++;
        else begin
          errs++;
          if (errs <= 4) $display("xdma: read beat %08h holds %032h", a, rdata);
        end
        rk++;
        if (rlast != (rk == rq[0].len)) begin
          errs++;
          if (errs <= 4) $display("xdma: RLAST %0d at beat %0d of %0d", rlast, rk, rq[0].len);
        end
        if (rlast) begin
          void'(rq.pop_front());
          rk = 0;
          r_out--;
        end
      end
      rready <= $urandom % 100 >= 32'(stall);
      done <= phase == 1 && nw == ntx && nr == ntx && b_out == 0 && r_out == 0;
    end
  end
  initial begin errs = 0; rchk = 0; nwb = 0; nb = 0; nrb = 0; end
endmodule
