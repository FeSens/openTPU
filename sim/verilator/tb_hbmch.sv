// Unit test of the WG targets' channel bridge (rtl/mem/otpu_hbm_ch.sv, with otpu_afifo and
// otpu_sfifo) in front of a model of one HBM AXI3 port (otpu_hbm_model: reads and writes in
// flight pass each other either way, as AXI allows). A random native master (the accelerator's
// side, core clock) drives the bridge; two unrelated clocks (+cp= / +ap=: half periods in ps;
// default 3750 / 1111, 133.33 / 450 MHz).
//
// The native master runs a program of random runs of 1 to LMAX beats over a window of +nb=N
// beats (reads or writes, +wpct=P percent writes; +ppct=P percent of the write beats partial: one
// byte, all but one, a range, none or random bytes written; +raw=P percent of the runs read back
// the last write run at once, +war=P percent write over the last read run at once). Commands and
// write data go out independently (random gaps: +gapc=P, +gapw=P), so a write's data comes before
// or after its command. It keeps room for its reads (+rbuf=N beats, drained with +mstall=P
// percent stalls) and checks every read beat against a byte shadow as of the read's place in the
// program (one master's commands run in order). It checks that n_wdone counts every write exactly
// once, never one whose command or data it has not yet given, and never one the memory has not
// landed (the model's wap: its W beats landed, in AW order; summed across the model's resets).
// +areset=N: reset the master and the bridge's core side at core cycle N for +arlen=L cycles
// (default 40), again every +arep=P cycles; the bytes of the writes then not yet counted become
// unknown (not checked) until written again. +hreset=N: the HBM's reset (arst, the model's too)
// alone at aclk cycle N for +hrlen=L cycles (default 2), again every +hrep=P cycles; the master
// then resets with the bridge's core side (crst) as well as with rst. +ntx=N runs.
// Throughput: +seq=1 (back-to-back LMAX-beat runs through the window, +wpct as given); the master
// prints its data beats per core cycle and the model its R beats per aclk cycle. +seq=2: the same
// sequence of beats in runs of random lengths and kinds (a read run starting where a write run
// ended, mid-block). +mw=N: the
// bridge's mwait. The bridge's n_err must stay clear. +cdc_skew=P: n_wdone's synchronizer sampling
// bits a source cycle apart (otpu_hbm_ch). WPOST 1: the master posts its write data on the
// bridge's n_wfree (a beat a cycle while n_wfree, registered, was at least 3 two cycles before:
// the strobes it does not count yet; WHB: a beat every two cycles while it was at least 2, the one
// beat it does not count yet at most, so a bridge offering a beat over u_wd's room fails), never
// waiting on n_wready; the bridge fails a posted write into a full u_wd. Prints "PASS" or the
// first mismatches.
// Posted reads (+ppost=P: P percent of the read runs, source 3): each beat takes the next of
// +nslot=N lane RAM entries (default 64, at most N outstanding, N up to 2^XSW: XSW the entry's
// bits, the board's log2 FIFO_DEPTH) and comes back on p_* (aclk) into
// the master's lane RAM model; the master takes p_cntg through two flip-flops into the core clock
// (as the adapter does) and at each step checks the oldest posted read: its entry written exactly
// K quarters since the read was given, the data as of its place in the program. Through a core
// reset the master takes the count as it is every reset cycle; a step after rst falls with no
// posted read given since is a failure (a pre-reset read's late count, or its quarter written
// into an entry reserved again). p_cntg must step by one gray code at a time (aclk), each step
// after a beat's last quarter (p_q's bit K - 1) is written, on its edge at the earliest. The other
// reads take sources 0-2, one per run. N: the bridge's pipeline (N + 1 stages each way), WH: W at
// half width there; AWD: the W FIFO's depth (AXI beats).
// Half beats (otpu_hbm_ch RHB / WHB, as otpu_native_dram's; the low half first, in n_rdata's,
// n_wdata's and n_wmask's low halves). RHB: each read beat comes as two halves on n_rvalid, the high
// one the cycle after the low one whatever the resets (the halves' order is a toggle on each side);
// a low half with no high half the next cycle, or a beat across the end of a reset, is a failure.
// WHB (with WPOST): the master posts a beat's low half on n_wfree as at WHB 0 and its high half the
// next cycle, the other bits random; a reset between them drops the beat.
module tb_hbmch #(
  parameter int CB = 128,
  parameter int ARD = 64,
  parameter int WOT = 16,
  parameter int NBK = 32,
  parameter int WDD = 32,
  parameter bit WPOST = 1'b0,
  parameter int N = 0,
  parameter bit WH = 1'b0,
  parameter int PST = 1,
  parameter int AWD = 32,
  parameter int XSW = 9,
  parameter bit RHB = 1'b0,
  parameter bit WHB = 1'b0,
  parameter bit CPY = 1'b0                     // the bridge's aclk fan-outs in copies (otpu_hbm_ch CPY)
);
  initial if (WHB && !WPOST) $fatal(1, "tb_hbmch: WHB needs WPOST");
  localparam int CAW = 34 - 5 - $clog2(CB);
  localparam int K = CB / 32;
  int cp = 3750, ap = 1111;
  logic clk = 1'b0, aclk = 1'b0;
  initial begin
    int seed;
    void'($value$plusargs("cp=%d", cp));
    void'($value$plusargs("ap=%d", ap));
    if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));
  end
  always #(cp) clk = ~clk;
  always #(ap) aclk = ~aclk;

  // resets
  logic rst = 1'b1, arst = 1'b1;
  longint ccyc = 0, acyc = 0;
  longint areset = -1, arep = 0, arlen = 40, hreset = -1, hrep = 0, hrlen = 2, hr_n = 0;
  int mw = 8;
  initial begin
    void'($value$plusargs("areset=%d", areset));
    void'($value$plusargs("arep=%d", arep));
    void'($value$plusargs("arlen=%d", arlen));
    void'($value$plusargs("hreset=%d", hreset));
    void'($value$plusargs("hrep=%d", hrep));
    void'($value$plusargs("hrlen=%d", hrlen));
    void'($value$plusargs("mw=%d", mw));
  end
  function automatic bit at(input longint c, input longint s, input longint p);
    return s > 0 && c >= s && (p > 0 ? (c - s) % p == 0 : c == s);
  endfunction
  always @(posedge aclk) begin
    acyc <= acyc + 1;
    if (acyc == 60) arst <= 1'b0;
    if (at(acyc, hreset, hrep)) begin arst <= 1'b1; hr_n = hrlen; end
    else if (hr_n > 0) begin hr_n--; if (hr_n == 0) arst <= 1'b0; end
  end
  always @(posedge clk) begin
    ccyc <= ccyc + 1;
    if (ccyc == 30) rst <= 1'b0;
    if (at(ccyc, areset, arep)) rst <= 1'b1;
    if (at(ccyc, areset + arlen, arep)) rst <= 1'b0;
  end

  logic ncv, ncr, ncwe, nwv, nwr, nrv, nerr;
  logic [CAW-1:0] nca;
  logic [1:0] ncsrc;
  logic [XSW-1:0] ncslot;
  logic [4:0][XSW-1:0] pslot;                   // (otpu_hbm_ch's copies)
  logic pwe;
  logic [1:0][K-1:0] pq;
  logic [255:0] pd;
  logic [XSW+1:0] pcntg;
  logic [CB*8-1:0] nwd, nrd;
  logic [CB-1:0] nwm;
  logic [15:0] nwdone;
  logic [$clog2(WDD):0] nwfree;
  logic awv, awr, wv, wr, wl, bv, br, arv, arr, rv, rr, rl;
  logic [33:0] awa, ara;
  logic [3:0] awl, arl;
  logic [255:0] wd, rd;
  logic [31:0] ws;
  logic [1:0] bre, rre;
  logic [3:0] ev;
  logic [$clog2(16 / K):0] ev_n;
  longint wap, wap_q = 0, wap_all = 0;      // the model's landed beats, and summed over its resets
  int     merr;
  logic   done, bad;

  tb_hbmch_nat #(.CB(CB), .CAW(CAW), .POST(WPOST), .XSW(XSW), .RHB(RHB), .WHB(WHB)) u_acc (
    .clk, .rst, .n_cvalid(ncv), .n_cready(ncr), .n_cwe(ncwe), .n_caddr(nca), .n_csrc(ncsrc),
    .n_cslot(ncslot), .n_wvalid(nwv), .n_wready(nwr), .n_wfree(8'(nwfree)), .n_wdata(nwd), .n_wmask(nwm),
    .n_rvalid(nrv), .n_rdata(nrd), .n_wdone(nwdone), .aclk, .p_we(pwe), .p_q(pq), .p_slot(pslot),
    .p_d(pd), .p_cntg(pcntg), .crst(u_ch.a_crst), .wap(wap_all), .done, .bad);

  otpu_hbm_ch #(.CB(CB), .CAW(CAW), .PC(5'd3), .ARD(ARD), .WOT(WOT), .NBK(NBK), .WDD(WDD),
                .WPOST(WPOST), .N(N), .WH(WH), .PST(PST), .AWD(AWD), .XSW(XSW), .RHB(RHB), .WHB(WHB),
                .CPY(CPY)) u_ch (
    .clk, .rst, .n_cvalid(ncv), .n_cready(ncr), .n_cwe(ncwe), .n_caddr(nca), .n_csrc(ncsrc),
    .n_cslot(ncslot), .n_wvalid(nwv), .n_wready(nwr), .n_wfree(nwfree), .n_wdata(nwd), .n_wmask(nwm),
    .n_rvalid(nrv), .n_rdata(nrd), .n_wdone(nwdone), .n_err(nerr),
    .p_we(pwe), .p_q(pq), .p_slot(pslot), .p_d(pd), .p_cntg(pcntg), .aclk, .arst, .mwait(6'(mw)),
    .m_awvalid(awv), .m_awready(awr), .m_awaddr(awa), .m_awlen(awl),
    .m_wvalid(wv), .m_wready(wr), .m_wdata(wd), .m_wstrb(ws), .m_wlast(wl),
    .m_bvalid(bv), .m_bready(br), .m_bresp(bre),
    .m_arvalid(arv), .m_arready(arr), .m_araddr(ara), .m_arlen(arl),
    .m_rvalid(rv), .m_rready(rr), .m_rdata(rd), .m_rresp(rre), .m_rlast(rl), .ev, .ev_n);

  otpu_hbm_model #(.BEATS(1 << 16), .PC(5'd3)) u_mem (
    .clk(aclk), .rst(arst), .dump(1'b0),
    .awvalid(awv), .awready(awr), .awid(6'd0), .awaddr(awa), .awlen(awl),
    .wvalid(wv), .wready(wr), .wdata(wd), .wstrb(ws), .wlast(wl),
    .bvalid(bv), .bready(br), .bid(), .bresp(bre),
    .arvalid(arv), .arready(arr), .arid(6'd0), .araddr(ara), .arlen(arl),
    .rvalid(rv), .rready(rr), .rid(), .rdata(rd), .rresp(rre), .rlast(rl), .wap, .nerr(merr));

  always @(posedge aclk) begin
    wap_all += (wap >= wap_q) ? wap - wap_q : wap;
    wap_q = wap;
  end

  // the bridge's issue counts (ev): bursts and their native beats, hazard and credit cycles
  longint n_ar = 0, n_arb = 0, n_aw = 0, n_awb = 0, n_hw = 0, n_cr = 0;
  always @(posedge aclk) begin
    if (ev[0]) begin n_ar++; n_arb += longint'(ev_n); end
    if (ev[1]) begin n_aw++; n_awb += longint'(ev_n); end
    n_cr += longint'(ev[2]);
    n_hw += longint'(ev[3]);
  end
  final $display("bridge: %0d read bursts (%.2f beats each), %0d write bursts (%.2f), %0d credit-held cycles, %0d hazard-held write cycles",
                 n_ar, real'(n_arb) / real'(n_ar > 0 ? n_ar : 1), n_aw, real'(n_awb) / real'(n_aw > 0 ? n_aw : 1),
                 n_cr, n_hw);

  // p_cntg: one gray code step at a time, each after a beat's last quarter (pc_pend: such beats
  // not yet counted)
  logic [XSW+1:0] pcg_q = '0;
  int pcg_err = 0, pc_pend = 0;
  function automatic logic [XSW+1:0] g2b(input logic [XSW+1:0] g);
    logic [XSW+1:0] b;
    b[XSW+1] = g[XSW+1];
    for (int i = XSW; i >= 0; i--) b[i] = b[i + 1] ^ g[i];
    return b;
  endfunction
  always @(posedge aclk) begin
    if (pcntg != pcg_q && g2b(pcntg) != g2b(pcg_q) + (XSW + 2)'(1)) begin
      if (pcg_err < 10) $display("ERROR p_cntg %h -> %h: not one gray code step", pcg_q, pcntg);
      pcg_err++;
    end
    if (pcntg != pcg_q) begin
      if (pc_pend == 0) begin
        if (pcg_err < 10) $display("ERROR p_cntg steps before a posted beat's last quarter is written");
        pcg_err++;
      end else pc_pend--;
    end
    if (pwe && pq[0][K-1]) pc_pend++;
    pcg_q <= pcntg;
  end

  longint tmax = 4000000;
  initial void'($value$plusargs("tmax=%d", tmax));
  always @(posedge clk) begin
    if (nerr) begin
      $display("FAIL n_err: the bridge saw a response it did not expect");
      $finish;
    end
    if (done) begin
      repeat (4) @(posedge clk);
      $display("%s cycles=%0d", (bad || merr != 0 || pcg_err != 0) ? "FAIL" : "PASS", ccyc);
      $finish;
    end
    if (ccyc > tmax) begin
      $display("TIMEOUT");
      $finish;
    end
  end
endmodule

// A random native master (the accelerator's side of otpu_hbm_ch; see tb_hbmch).
module tb_hbmch_nat #(
  parameter int CB = 128,
  parameter int CAW = 22,
  parameter int LMAX = 32,
  parameter string NAME = "acc",
  parameter bit POST = 1'b0,                   // write data posted on n_wfree (tb_hbmch WPOST)
  parameter int XSW = 9,                       // a posted read's lane RAM entry's bits
  parameter bit RHB = 1'b0,                    // read data in half beats (tb_hbmch RHB)
  parameter bit WHB = 1'b0                     // posted write data in half beats (tb_hbmch WHB)
) (
  input  logic            clk,
  input  logic            rst,
  output logic            n_cvalid,
  input  logic            n_cready,
  output logic            n_cwe,
  output logic [CAW-1:0]  n_caddr,
  output logic [1:0]      n_csrc,
  output logic [XSW-1:0]  n_cslot,
  output logic            n_wvalid,
  input  logic            n_wready,
  input  logic [7:0]      n_wfree,
  output logic [CB*8-1:0] n_wdata,
  output logic [CB-1:0]   n_wmask,
  input  logic            n_rvalid,
  input  logic [CB*8-1:0] n_rdata,
  input  logic [15:0]     n_wdone,
  input  logic            aclk,                // posted reads' beats (otpu_hbm_ch p_*)
  input  logic            p_we,
  input  logic [1:0][CB/32-1:0] p_q,         // one-hot with p_we, else 0; two copies
  input  logic [4:0][XSW-1:0] p_slot,            // five copies
  input  logic [255:0]    p_d,
  input  logic [XSW+1:0]  p_cntg,
  input  logic            crst,                // the bridge's core side in reset
  input  longint          wap,                 // the model's W beats landed
  output logic            done,
  output logic            bad
);
  localparam int K = CB / 32;
  localparam int NBM = 8192;                   // the most beats a window may have
  logic [7:0] sh [NBM * CB];
  bit         unk [NBM * CB];
  typedef struct { bit we; bit post; logic [1:0] src; int b; logic [CB*8-1:0] d; logic [CB-1:0] m; } op_t;
  typedef struct { logic [CB*8-1:0] d; logic [CB-1:0] m; } wd_t;
  typedef struct { int b; logic [CB-1:0] m; } pw_t;           // a write not yet counted
  typedef struct { int b; int slot; longint q0; logic [CB*8-1:0] d; logic [CB-1:0] m; } pr_t;  // posted
  op_t cq [$], rq [$];
  pr_t pq [$];
  wd_t dq [$];
  pw_t pw [$];
  int ntx = 400, issued = 0, nerr = 0, nb = 1024;
  int gapw = 20, gapc = 10, mstall = 30, wpct = 50, seq = 0, ppct = 25, raw = 20, war = 10;
  int rbuf = 96;
  int ppost = 0, nslot = 64, pout = 0, nslt = 0, drv_slot = 0;
  longint npb = 0;
  int rres = 0, occ = 0, nextb = 0, lw_b0 = -1, lw_n = 0, lr_b0 = -1, lr_n = 0;
  longint wacc = 0, wdat = 0, wdone_tot = 0, nrb = 0, nwb = 0, npart = 0, wap0 = 0;
  logic [15:0] wd_prev = '0;
  longint c0 = -1, c1 = 0, cyc = 0;
  bit crst_q = 1'b1;
  bit w_ok = 1'b0;                             // POST: n_wfree >= 3 (WHB: 2), as of the last edge
  bit hrm = 1'b0;                              // +hreset: reset with the bridge's core side too
  bit rv;                                      // a read beat whole this cycle (RHB: its high half)
  logic [CB*8-1:0] rdat;
  bit rh = 1'b0, rh_rst = 1'b0;                // RHB: this half is a high one; its low one in a reset
  logic [CB*4-1:0] rh_lo;
  bit wh = 1'b0;                               // WHB: the half on the bus is a high one

  initial begin
    void'($value$plusargs("ntx=%d", ntx));
    void'($value$plusargs("nb=%d", nb));
    void'($value$plusargs("mstall=%d", mstall));
    void'($value$plusargs("wpct=%d", wpct));
    void'($value$plusargs("seq=%d", seq));
    void'($value$plusargs("gapw=%d", gapw));
    void'($value$plusargs("gapc=%d", gapc));
    void'($value$plusargs("ppct=%d", ppct));
    void'($value$plusargs("raw=%d", raw));
    void'($value$plusargs("war=%d", war));
    void'($value$plusargs("rbuf=%d", rbuf));
    void'($value$plusargs("ppost=%d", ppost));
    void'($value$plusargs("nslot=%d", nslot));
    hrm = $test$plusargs("hreset");
    if (nslot > (1 << XSW)) nslot = 1 << XSW;
    if (nb > NBM) nb = NBM;
    for (int i = 0; i < NBM * CB; i++) begin sh[i] = 8'h00; unk[i] = 1'b0; end
    n_cvalid = 1'b0; n_wvalid = 1'b0;
  end

  // a partial beat's byte enables
  function automatic logic [CB-1:0] pmask();
    logic [CB-1:0] m;
    int a, b;
    case ($urandom % 6)
      0: m = CB'(1) << ($urandom % CB);
      1: m = ~(CB'(1) << ($urandom % CB));
      2: begin
        a = $urandom % CB; b = $urandom % CB;
        if (a > b) begin int t; t = a; a = b; b = t; end
        m = '0;
        for (int k = a; k <= b; k++) m[k] = 1'b1;
        if (m == '1) m[CB-1] = 1'b0;
      end
      3: m = '0;
      default: begin
        for (int k = 0; k < CB; k += 32) m[k +: 32] = $urandom;
        if (m == '1) m[7] = 1'b0;
      end
    endcase
    return m;
  endfunction

  // the other half's bits (WHB)
  function automatic logic [CB*4-1:0] junk();
    logic [CB*4-1:0] j;
    for (int k = 0; k < CB * 4; k += 32) j[k +: 32] = $urandom;
    return j;
  endfunction

  // posted reads: the lane RAM (aclk, a quarter at a time) and each entry's quarters written; the
  // count through two flip-flops into clk
  logic [CB*8-1:0] lram [1 << XSW];
  longint          lqn [1 << XSW];
  initial for (int i = 0; i < (1 << XSW); i++) lqn[i] = 0;
  always @(posedge aclk) begin
    if (p_we ? !$onehot(p_q[0]) : p_q[0] != '0)
      $fatal(1, "tb_hbmch: p_q %b with p_we %0d (one-hot with it, else 0)", p_q[0], p_we);
    if (p_q[1] != p_q[0] || p_slot != {5{p_slot[0]}})
      $fatal(1, "tb_hbmch: p_q / p_slot's copies differ (%h, %h)", p_q, p_slot);
    if (p_we) begin
      lram[p_slot[0]][256 * $clog2(p_q[0]) +: 256] <= p_d;
      lqn[p_slot[0]] <= lqn[p_slot[0]] + 1;
    end
  end
  logic [XSW+1:0] pc_s1 = '0, pc_s2 = '0, pc_b = '0;
  always @(posedge clk) begin pc_s1 <= p_cntg; pc_s2 <= pc_s1; end
  function automatic logic [XSW+1:0] pg2b(input logic [XSW+1:0] g);
    logic [XSW+1:0] b;
    b[XSW+1] = g[XSW+1];
    for (int i = XSW; i >= 0; i--) b[i] = b[i + 1] ^ g[i];
    return b;
  endfunction

  always @(posedge clk) begin
    cyc++;
    // RHB: a read beat's halves, the high one the cycle after the low one, in resets or not
    rv = n_rvalid; rdat = n_rdata;
    if (RHB) begin
      if (rh && !n_rvalid) begin
        if (nerr < 20) $display("ERROR %s: a read beat's low half without its high half the next cycle", NAME);
        nerr++;
      end
      if (rh && n_rvalid && rh_rst && !(rst || (hrm && crst))) begin
        if (nerr < 20) $display("ERROR %s: a read beat's low half in a reset, its high half after it", NAME);
        nerr++;
      end
      rv = rh && n_rvalid;
      rdat = {n_rdata[CB*4-1:0], rh_lo};
      if (!rh) begin rh_lo = n_rdata[CB*4-1:0]; rh_rst = rst || (hrm && crst); end
      rh = !rh && n_rvalid;
    end
    // the model's landed beats when the bridge's core side leaves reset: n_wdone counts from 0 then
    if (crst_q && !crst) wap0 = wap;
    crst_q = crst;
    if (rst || (hrm && crst)) begin
      // writes not counted may or may not land: their bytes become unknown
      foreach (pw[j]) for (int k = 0; k < CB; k++) if (pw[j].m[k]) unk[CB * pw[j].b + k] = 1'b1;
      cq.delete(); dq.delete(); rq.delete(); pw.delete(); pq.delete();
      rres = 0; occ = 0; wacc = 0; wdat = 0; wdone_tot = 0; wd_prev = '0; pout = 0;
      pc_b = pg2b(pc_s2);                      // the adapter takes the count every reset cycle
      n_cvalid <= 1'b0; n_wvalid <= 1'b0; w_ok = 1'b0; wh = 1'b0;
    end else begin
      // ---- n_wdone: never ahead of the writes given (command and data) nor of the memory's
      // landed beats, each counted once
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
          for (int j = 0; j < int'(d); j++) void'(pw.pop_front());
          wdone_tot += longint'(d);
          nwb += longint'(d);
          c1 = cyc;
          if (K * wdone_tot > wap - wap0) begin
            if (nerr < 20)
              $display("ERROR %s: n_wdone counts %0d beats, the memory has landed %0d", NAME,
                       wdone_tot, (wap - wap0) / K);
            nerr++;
          end
        end
        wd_prev = n_wdone;
      end
      // ---- read data, in command order, into the room kept for it
      if (rv) begin
        if (rq.size() == 0) begin $display("ERROR %s: read data without a read", NAME); nerr++; end
        else begin
          for (int k = 0; k < CB; k++)
            if (rq[0].m[k] && rdat[8 * k +: 8] !== rq[0].d[8 * k +: 8]) begin
              if (nerr < 20)
                $display("ERROR %s: beat %0d byte %0d: %h, expected %h", NAME, rq[0].b, k,
                         rdat[8 * k +: 8], rq[0].d[8 * k +: 8]);
              nerr++;
            end
          void'(rq.pop_front());
          rres--; occ++; nrb++; c1 = cyc;
        end
      end
      if (occ > 0 && ($urandom % 100) >= mstall) occ--;
      // ---- posted reads, in command order, at each step of the count
      begin
        logic [XSW+1:0] pd_;
        pd_ = pg2b(pc_s2) - pc_b;
        for (int j = 0; j < int'(pd_); j++) begin
          if (pq.size() == 0) begin
            if (nerr < 20) $display("ERROR %s: p_cnt steps with no posted read given", NAME);
            nerr++;
          end else begin
            if (lqn[pq[0].slot] - pq[0].q0 != longint'(K)) begin
              if (nerr < 20)
                $display("ERROR %s: posted beat %0d (entry %0d) counted with %0d quarters written", NAME,
                         pq[0].b, pq[0].slot, lqn[pq[0].slot] - pq[0].q0);
              nerr++;
            end
            for (int k = 0; k < CB; k++)
              if (pq[0].m[k] && lram[pq[0].slot][8 * k +: 8] !== pq[0].d[8 * k +: 8]) begin
                if (nerr < 20)
                  $display("ERROR %s: posted beat %0d (entry %0d) byte %0d: %h, expected %h", NAME,
                           pq[0].b, pq[0].slot, k, lram[pq[0].slot][8 * k +: 8], pq[0].d[8 * k +: 8]);
                nerr++;
              end
            void'(pq.pop_front());
            pout--; npb++; c1 = cyc;
          end
        end
        pc_b = pg2b(pc_s2);
      end
      // ---- write data, in write order, independent of the commands
      if (POST && WHB) begin                   // posted halves: the high one the next cycle
        if (n_wvalid && wh) begin void'(dq.pop_front()); wdat++; end
        if (n_wvalid && !wh) begin
          n_wvalid <= 1'b1; wh = 1'b1;
          n_wdata <= {junk(), dq[0].d[CB*8-1:CB*4]}; n_wmask <= {(CB / 2)'(junk()), dq[0].m[CB-1:CB/2]};
        end else if (dq.size() != 0 && w_ok && ($urandom % 100) >= gapw) begin
          n_wvalid <= 1'b1; wh = 1'b0;
          n_wdata <= {junk(), dq[0].d[CB*4-1:0]}; n_wmask <= {(CB / 2)'(junk()), dq[0].m[CB/2-1:0]};
        end else begin n_wvalid <= 1'b0; wh = 1'b0; end
        w_ok = n_wfree >= 8'd2;
      end else if (POST) begin                 // posted: a beat goes when it is driven
        if (n_wvalid) begin void'(dq.pop_front()); wdat++; end
        if (dq.size() != 0 && w_ok && ($urandom % 100) >= gapw) begin
          n_wvalid <= 1'b1; n_wdata <= dq[0].d; n_wmask <= dq[0].m;
        end else n_wvalid <= 1'b0;
        w_ok = n_wfree >= 8'd3;
      end else begin
        if (n_wvalid && n_wready) begin void'(dq.pop_front()); wdat++; end
        if (!n_wvalid || n_wready) begin
          if (dq.size() != 0 && ($urandom % 100) >= gapw) begin
            n_wvalid <= 1'b1; n_wdata <= dq[0].d; n_wmask <= dq[0].m;
          end else n_wvalid <= 1'b0;
        end
      end
      // ---- commands; a read only with room for its data
      if (n_cvalid && n_cready) begin
        if (cq[0].we) wacc++;
        else if (cq[0].post) pq.push_back('{cq[0].b, drv_slot, lqn[drv_slot], cq[0].d, cq[0].m});
        else rq.push_back(cq[0]);
        void'(cq.pop_front());
        if (c0 < 0) c0 = cyc;
      end
      if (!n_cvalid || n_cready) begin
        if (cq.size() != 0 && (seq != 0 || ($urandom % 100) >= gapc) &&
            (cq[0].we || (cq[0].post ? pout < nslot : rres + occ < rbuf))) begin
          n_cvalid <= 1'b1; n_cwe <= cq[0].we; n_caddr <= CAW'(cq[0].b); n_csrc <= cq[0].src;
          if (cq[0].post) begin
            drv_slot = nslt; nslt = (nslt + 1) % nslot; pout++;
            n_cslot <= XSW'(drv_slot);
          end else n_cslot <= XSW'($urandom);
          if (!cq[0].we && !cq[0].post) rres++;
        end else n_cvalid <= 1'b0;
      end
      // ---- the program: a run of beats
      if (issued < ntx && cq.size() < 8) begin
        int n, b0;
        bit wr_, rw, wa, po;
        logic [1:0] src;
        rw  = seq == 0 && lw_b0 >= 0 && ($urandom % 100) < raw;
        wa  = seq == 0 && !rw && lr_b0 >= 0 && ($urandom % 100) < war;
        wr_ = wa || (!rw && ($urandom % 100) < wpct);
        n   = seq == 1 ? LMAX : rw ? lw_n : wa ? lr_n : 1 + $urandom % LMAX;
        b0  = seq != 0 ? nextb : rw ? lw_b0 : wa ? lr_b0 : $urandom % nb;
        if (b0 + n > nb) n = nb - b0;
        nextb = (b0 + n) % nb;
        if (wr_) begin lw_b0 = b0; lw_n = n; end else begin lr_b0 = b0; lr_n = n; end
        po  = !wr_ && ($urandom % 100) < ppost;
        src = wr_ ? 2'd0 : po ? 2'd3 : 2'($urandom % 3);
        for (int j = 0; j < n; j++) begin
          op_t o;
          int b;
          b = b0 + j;
          o.we = wr_; o.post = po; o.src = src; o.b = b;
          if (wr_) begin
            o.m = (seq != 1 && ($urandom % 100) < ppct) ? pmask() : '1;
            for (int k = 0; k < CB; k++) begin
              o.d[8 * k +: 8] = 8'($urandom);
              if (o.m[k]) begin sh[CB * b + k] = o.d[8 * k +: 8]; unk[CB * b + k] = 1'b0; end
            end
            if (o.m != '1) npart++;
            dq.push_back('{o.d, o.m});
            pw.push_back('{b, o.m});
          end else begin
            for (int k = 0; k < CB; k++) begin
              o.d[8 * k +: 8] = sh[CB * b + k];
              o.m[k] = !unk[CB * b + k];
            end
          end
          cq.push_back(o);
        end
        issued++;
      end
    end
  end
  assign done = (issued >= ntx && cq.size() == 0 && dq.size() == 0 && rq.size() == 0 && pq.size() == 0 && !n_cvalid &&
                 !n_wvalid && wdone_tot == wacc && wacc == wdat) || nerr != 0;
  assign bad  = nerr != 0;

  final
    $display("%s: %0d runs, %0d read beats, %0d posted, %0d write beats counted (%0d partial), %0d errors, %.3f beats per cycle",
             NAME, issued, nrb, npb, nwb, npart, nerr, real'(nrb + npb + nwb) / real'(c1 - c0 + 1));
endmodule
