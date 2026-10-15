// Unit test of otpu_pcie_mm (rtl/shell: LitePCIe's DMA streams onto the card's memory port) through
// the real memory path behind it: otpu_dma_split (XREG 2: its register slices, as on the card), one
// otpu_mem_ch per channel and a model of each channel's LiteDRAM native ports (otpu_ldn_model), at
// the card's clocks (PCIe user clock 250 MHz, controller and core 133.33 MHz: +xp= / +up= / +cp=,
// half periods). The accelerator side of the bridges is idle.
//
// The host (this bench) runs +nop=N operations in phases: a run of host->card commands (random
// card addresses and lengths in 16-byte units, from any 16-byte offset, +al64=P percent 64-byte
// aligned; +fill=P percent FILL commands), several in flight, then a run of card->host commands
// over random ranges, also several in flight, while the next run of writes goes to the other
// half of the windows (+mix=1). The H2C stream carries a hash of its byte position, with random
// valid gaps (+hgap=P percent); the C2H stream is held back at random (+cstall=P percent). A byte
// shadow of each channel's window checks every C2H beat (as of the read's issue: no write in
// flight touches a range being read) and the last flag on each command's final beat;
// h2c_done must pulse once per command, in order, never before the command's last write
// response. At the end err must be clear and the engine idle. +zero=P percent zero-length
// commands of either kind.
// +seq=1: throughput, back-to-back 64 KiB commands with the streams never held: prints H2C and
// C2H beats per xclk cycle (16 bytes a beat: 1.0 = 4 GB/s at 250 MHz).
// Prints "PASS" or the first mismatches.
module tb_pcie_mm #(
  parameter int XREG = 2
);
  int cp = 375, up = 375, xp = 200;
  logic clk = 1'b0, uclk = 1'b0, xclk = 1'b0;
  int nop = 400, al64 = 50, fillp = 10, hgap = 20, cstall = 20, zero = 3, seq = 0, mix = 1;
  initial begin
    int seed;
    void'($value$plusargs("cp=%d", cp));
    void'($value$plusargs("up=%d", up));
    void'($value$plusargs("xp=%d", xp));
    void'($value$plusargs("nop=%d", nop));
    void'($value$plusargs("al64=%d", al64));
    void'($value$plusargs("fill=%d", fillp));
    void'($value$plusargs("hgap=%d", hgap));
    void'($value$plusargs("cstall=%d", cstall));
    void'($value$plusargs("zero=%d", zero));
    void'($value$plusargs("seq=%d", seq));
    void'($value$plusargs("mix=%d", mix));
    if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));
  end
  always #(cp) clk = ~clk;
  always #(up) uclk = ~uclk;
  always #(xp + 1) xclk = ~xclk;

  localparam int BEATS = 1 << 13;                 // 512 KiB per channel model
  localparam int WIN = BEATS * 64;

  logic rst = 1'b1, urst = 1'b1, xrst = 1'b1;
  longint xcyc = 0;
  int ucyc = 0, ccyc = 0;
  always @(posedge uclk) begin ucyc <= ucyc + 1; if (ucyc == 20) urst <= 1'b0; end
  always @(posedge clk) begin ccyc <= ccyc + 1; if (ccyc == 30) rst <= 1'b0; end
  always @(posedge xclk) begin xcyc <= xcyc + 1; if (xcyc == 40) xrst <= 1'b0; end

  // ---- the DUT, the split, the bridges and the controllers
  logic hcv, hcr, hdv, hdr, hdone, ccv_, ccr_, cdv, cdr, cdl, err;
  logic [31:0] hca, cca_, status;
  logic [23:0] hcl, ccl;
  logic [7:0] hcf, ccf;
  logic [127:0] hdd, cdd;
  logic xawv, xawr, xwv, xwr, xwl, xbv, xbr, xarv, xarr, xrv, xrr, xrl;
  logic [3:0] xawi, xbi, xari, xri;
  logic [31:0] xawa, xara;
  logic [7:0] xawl, xarl;
  logic [127:0] xwd, xrd;
  logic [15:0] xws;
  logic [1:0] xbre, xrre;

  otpu_pcie_mm u_mm (
    .clk(xclk), .rst(xrst),
    .h2c_cmd_valid(hcv), .h2c_cmd_ready(hcr), .h2c_cmd_addr(hca), .h2c_cmd_len(hcl),
    .h2c_cmd_flags(hcf), .h2c_dat_valid(hdv), .h2c_dat_ready(hdr), .h2c_dat_data(hdd),
    .h2c_done(hdone),
    .c2h_cmd_valid(ccv_), .c2h_cmd_ready(ccr_), .c2h_cmd_addr(cca_), .c2h_cmd_len(ccl),
    .c2h_cmd_flags(ccf), .c2h_dat_valid(cdv), .c2h_dat_ready(cdr), .c2h_dat_data(cdd),
    .c2h_dat_last(cdl), .status, .err,
    .m_awvalid(xawv), .m_awready(xawr), .m_awid(xawi), .m_awaddr(xawa), .m_awlen(xawl),
    .m_wvalid(xwv), .m_wready(xwr), .m_wdata(xwd), .m_wstrb(xws), .m_wlast(xwl),
    .m_bvalid(xbv), .m_bready(xbr), .m_bid(xbi), .m_bresp(xbre),
    .m_arvalid(xarv), .m_arready(xarr), .m_arid(xari), .m_araddr(xara), .m_arlen(xarl),
    .m_rvalid(xrv), .m_rready(xrr), .m_rid(xri), .m_rdata(xrd), .m_rresp(xrre), .m_rlast(xrl));

  logic [1:0] cawv, cawr, cwvx, cwrx, cwl, cbv, cbr, carv, carr, crvx, crrx, crl;
  logic [1:0][3:0] cawi, cari, cbi, cri;
  logic [1:0][31:0] cawa, cara;
  logic [1:0][7:0] cawl, carl;
  logic [1:0][1:0] cbre, crre;
  logic [1:0][127:0] cwdx, crdx;
  logic [1:0][15:0] cws;
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

  logic [1:0][1:0] ccv, ccr, ccwe, cwv, cwr, crv;
  logic [1:0][1:0][24:0] cca;
  logic [1:0][1:0][511:0] cwd, crd;
  logic [1:0][1:0][63:0] cwe;
  logic [1:0] nerr;
  for (genvar c = 0; c < 2; c++) begin : g_ch
    logic ncr, nwr, nrv;
    logic [511:0] nrd;
    logic [15:0] nwdone;
    otpu_mem_ch #(.XIDW(4)) u_ch (
      .clk, .rst,
      .n_cvalid(1'b0), .n_cready(ncr), .n_cwe(1'b0), .n_caddr('0),
      .n_wvalid(1'b0), .n_wready(nwr), .n_wdata('0), .n_wmask('0),
      .n_rvalid(nrv), .n_rdata(nrd), .n_wdone(nwdone), .n_err(nerr[c]),
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
      .c_wdata_we(cwe[c]), .c_rdata_valid(crv[c]), .c_rdata_data(crd[c]), .dump(1'b0));
  end

  // ---- the host
  byte unsigned shadow [2][WIN];
  initial foreach (shadow[c, i]) shadow[c][i] = 8'h00;
  function automatic byte unsigned hb(input longint pos);
    logic [31:0] h;
    h = 32'(pos) * 32'h9E3779B1 ^ 32'(pos >> 32) * 32'h85EBCA77;
    h = h ^ (h >> 15); h = h * 32'h2C1B3C6D; h = h ^ (h >> 13);
    return h[7:0];
  endfunction

  typedef struct { logic [31:0] addr; int len; bit fill; } op_t;
  op_t hq [$];                                    // H2C commands to give
  op_t cq [$];                                    // C2H commands to give
  int  h_given = 0, h_done_n = 0;                 // H2C commands given / done
  int  c_given = 0, c_done_n = 0;                 // C2H commands given / fully received
  longint hpos = 0;                               // H2C stream bytes queued
  longint hsent = 0;                              // H2C stream bytes given
  byte unsigned cexp [$];                         // C2H bytes expected, in order
  bit  clast [$];                                 // ... and whether each 16-byte beat is a last
  int  errors = 0;
  // bytes each given H2C command has written to the shadow (applied at its done)
  op_t hpend [$];
  longint hpend_pos [$];

  function automatic op_t rand_op(input int half, input int maxlen);
    op_t o;
    int c, len, off, lo, hi;
    c = $urandom % 2;
    len = 16 * (1 + $urandom % (maxlen / 16));
    if (($urandom % 100) < al64) len = (len + 63) / 64 * 64;
    if (($urandom % 100) < zero) len = 0;
    lo = half < 0 ? 0 : half * (WIN / 2);
    hi = half < 0 ? WIN : lo + WIN / 2;
    off = lo + 16 * ($urandom % ((hi - lo - len) / 16 + 1));
    if (($urandom % 100) < al64) off = off / 64 * 64;
    if (off + len > hi) off = hi - len;
    o.addr = {1'(c), 31'(off)};
    o.len = len;
    o.fill = (($urandom % 100) < fillp);
    return o;
  endfunction

  // H2C commands and stream
  op_t hcur;
  longint hnext = 0;                              // next stream byte to put on the bus
  always @(posedge xclk) begin
    if (xrst) begin
      hcv <= 1'b0; hdv <= 1'b0;
    end else begin
      if (hcv && hcr) begin hcv <= 1'b0; h_given++; end
      if ((!hcv || hcr) && hq.size() > 0 && (seq != 0 || ($urandom % 4) == 0)) begin
        hcur = hq.pop_front();
        hcv <= 1'b1; hca <= hcur.addr; hcl <= 24'(hcur.len); hcf <= {7'd0, hcur.fill};
        hpend.push_back(hcur);
        hpend_pos.push_back(hpos);
        if (!hcur.fill) hpos += hcur.len;
      end
      // the stream: the bytes of the commands queued, in order
      if (hdv && hdr) begin hdv <= 1'b0; hsent += 16; end
      if ((!hdv || hdr) && hnext < hpos && (seq != 0 || ($urandom % 100) >= hgap)) begin
        for (int k = 0; k < 16; k++) hdd[8 * k +: 8] <= hb(hnext + k);
        hdv <= 1'b1;
        hnext += 16;
      end
    end
  end

  // H2C done: apply the command's bytes to the shadow (none of them is being read)
  always @(posedge xclk) if (!xrst && hdone) begin
    op_t o;
    longint p;
    if (hpend.size() == 0) begin
      $display("FAIL: h2c_done with no command given"); errors++;
    end else begin
      o = hpend.pop_front();
      p = hpend_pos.pop_front();
      for (int i = 0; i < o.len; i++)
        shadow[o.addr[31]][o.addr[30:0] + i] = o.fill ? 8'h00 : hb(p + i);
      h_done_n++;
    end
  end

  // C2H commands and stream
  op_t ccur;
  always @(posedge xclk) begin
    if (xrst) begin
      ccv_ <= 1'b0; cdr <= 1'b0;
    end else begin
      if (ccv_ && ccr_) begin ccv_ <= 1'b0; c_given++; end
      if (!(ccv_ && !ccr_) && cq.size() > 0 && (seq != 0 || ($urandom % 4) == 0)) begin
        ccur = cq.pop_front();
        ccv_ <= 1'b1; cca_ <= ccur.addr; ccl <= 24'(ccur.len); ccf <= '0;
        for (int i = 0; i < ccur.len; i++) cexp.push_back(shadow[ccur.addr[31]][ccur.addr[30:0] + i]);
        for (int b = 0; b < ccur.len / 16; b++) clast.push_back(b == ccur.len / 16 - 1);
        if (ccur.len == 0) c_done_n++;
      end
      cdr <= seq != 0 || ($urandom % 100) >= cstall;
      if (cdv && cdr) begin
        bit l;
        if (cexp.size() < 16) begin
          $display("FAIL: C2H beat with no read given"); errors++;
        end else begin
          for (int k = 0; k < 16; k++) begin
            byte unsigned e;
            e = cexp.pop_front();
            if (cdd[8 * k +: 8] != e && errors < 20) begin
              $display("FAIL: C2H byte %0d of a beat: %h, expected %h (cycle %0d)", k, cdd[8 * k +: 8], e, xcyc);
              errors++;
            end
          end
          l = clast.pop_front();
          if (cdl != l) begin $display("FAIL: C2H last %0d, expected %0d", cdl, l); errors++; end
          if (l) c_done_n++;
        end
      end
    end
  end

  // the phases
  int nh = 0, nc = 0;
  longint t0, th, tc;
  initial begin
    @(negedge xrst);
    repeat (50) @(posedge xclk);
    if (seq != 0) begin
      op_t o;
      for (int i = 0; i < 8; i++) begin
        o.addr = {1'(i % 2), 31'((i / 2) * 65536)}; o.len = 65536; o.fill = 0;
        hq.push_back(o);
      end
      t0 = xcyc;
      wait (h_done_n == 8);
      th = xcyc - t0;
      for (int i = 0; i < 8; i++) begin
        o.addr = {1'(i % 2), 31'((i / 2) * 65536)}; o.len = 65536; o.fill = 0;
        cq.push_back(o);
      end
      t0 = xcyc;
      wait (c_done_n == 8);
      tc = xcyc - t0;
      $display("seq: H2C %.3f beats/cycle (%0d cycles), C2H %.3f beats/cycle (%0d cycles)",
               8.0 * 4096 / th, th, 8.0 * 4096 / tc, tc);
    end else begin
      int half;
      half = 0;
      while (nh + nc < nop) begin
        int k;
        // writes to one half (both halves when not mixing)
        k = 1 + $urandom % 8;
        for (int i = 0; i < k; i++) begin hq.push_back(rand_op(mix ? half : -1, 8192)); nh++; end
        // reads of the other half, while those writes run
        if (mix) begin
          k = 1 + $urandom % 8;
          for (int i = 0; i < k; i++) begin cq.push_back(rand_op(1 - half, 8192)); nc++; end
        end
        wait (hq.size() == 0 && cq.size() == 0 && !hcv && !ccv_ && h_done_n == nh && c_done_n == nc);
        if (!mix) begin
          k = 1 + $urandom % 8;
          for (int i = 0; i < k; i++) begin cq.push_back(rand_op(-1, 8192)); nc++; end
          wait (cq.size() == 0 && !ccv_ && c_done_n == nc);
        end
        half = 1 - half;
      end
    end
    repeat (200) @(posedge xclk);
    if (err) begin $display("FAIL: err set, status %h", status); errors++; end
    if (status[1:0] != 0) begin $display("FAIL: not idle, status %h", status); errors++; end
    if (cexp.size() != 0) begin $display("FAIL: %0d C2H bytes never came", cexp.size()); errors++; end
    if (nerr != 0) begin $display("FAIL: a bridge's n_err"); errors++; end
    if (errors == 0) $display("PASS (%0d host->card, %0d card->host commands)", h_done_n, c_done_n);
    $finish;
  end
  initial begin
    longint lim;
    lim = 64'd20_000_000;
    #(lim * 400);
    $display("TIMEOUT (h2c %0d/%0d given %0d, c2h %0d/%0d, status %h)", h_done_n, nh, h_given, c_done_n, nc, status);
    $finish;
  end
endmodule
