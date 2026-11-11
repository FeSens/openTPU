// tb_migpair: otpu_mig_pair (one DIMM's NC virtual channels, 2 or 4, onto its 512-bit AXI4 port)
// between NC masters with otpu_hbm_ch's AXI side and a model of the DIMM's port (tests/
// test_migpair.py; its sources include otpu_dimm_model.sv at any DM).
//
// The masters (one per bridge port): bursts of 4, 8, 12 or 16 256-bit beats (whole 128-byte
// channel beats) inside one aligned 512-byte block of the channel (+blocks=N of them: block k at
// k x 512, or with +hi at (k >> 4) x 512 + (k & 15) x 2^(CL-4), the channel's top four address
// bits in use, so a map that dropped or folded them would alias blocks), reads or
// writes (+wpct=P percent writes; +ppct=P percent of the beats with random strobes), at most +ot=N
// bursts in flight; a burst never goes while one of the other kind is in flight to its block
// (otpu_hbm_ch's hazard rule), so a read beat is checked against the channel's bytes (shadow) as
// they are when it comes back. AW before its W, the W beats with gaps (+gapw=P percent); R and B
// always taken. +seq: reads only, block after block, 16-beat bursts (the decode's stream: the
// throughput run).
//
// The model (512-bit AXI4, IDs): a memory of 512-bit beats; AR / AW queues (64 / 32 deep); a
// read's beats go +lat=N cycles after its AR at the earliest (+jit=N more at random), in AR order
// (allowed between IDs), an R beat in at most +rbw=P percent of the cycles; ARREADY, AWREADY and
// WREADY low in +st=P percent of the cycles; a write lands beat by beat, its B +wlat=N cycles
// after its last, in AW order. Checks each AR / AW: 64-byte aligned, inside 4 KiB, the ID's top
// log2 NC bits (the bridge) the address's bits from 12 (its channel), the ID's other bits 0; WLAST
// on a burst's last beat.
//
// The end: after +n=N bursts a master and nothing in flight, every byte either master wrote is
// compared with the model's memory at the DIMM address the pair's map gives ({o[CL-1:12], s,
// o[11:0]}, s log2 NC bits, computed here as arithmetic); then "PASS cycles C rbeats R wbeats W ...
// rrate X" (X:
// the model's R beats per cycle between its first and last), or FAIL / TIMEOUT (+maxc=N cycles).
//
// Every output toward the pair is a register, its next value computed at the clock edge from the
// state after that edge's handshakes (no race with the pair's registers). CL: the pair's channel
// byte address bits (29: 512 MiB; 32: 4 GiB, the 32 GB card; 33: 8 GiB).
//
// DM 1: the DIMM's port is otpu_dimm_model instead (the board model's: DAW CL + log2 NC, its first DMB
// beats a channel in its array, the rest in its sparse store; its own +dimm_* / +mc_* timing,
// +dimm_mc=1 the MIG's, and +dir=<d> images; +lat, +jit, +rbw, +st and +wlat do not apply, nor
// the AR / AW checks), the end's comparison read from its memory after its dump (which lands the
// writes still in its controller); an error of the model's is a FAIL.
//
// BHW (otpu_mig_pair's bank-half window; BHB its bank-high bit): the pair's map is the window's
// swap of {o[CL-1:12], s, o[11:0]} (computed here too: bhw), DM 1's model takes the same.
//
// HM 1: otpu_mig_hmux between the pair and the DIMM's port, its host side driven by a host master
// (the block design's M_AXI_H<d>: 512-bit beats, no IDs): +hn=N bursts (default 1000; 0: the host
// idle) of 1 .. +hmaxb=N beats (64) inside one 4 KiB block of its region, the DIMM's top +hblk=N
// blocks (16: no bridge's block there), reads or writes (+hwpct=P percent writes, 50; +hppct=P
// percent of the beats with random strobes, 30), at most +hot=N in flight (8), the masters' hazard
// rule per block, so its reads are checked against its bytes as they are; its W beats queued as it
// offers the burst (W may come before its AW, as a SmartConnect's), with gaps (+hgapw=P, 10);
// RREADY and BREADY low in +hst=P percent of the cycles (10). The model takes the host's ID
// (otpu_mig_hmux HB: {0, 1, 0, 0} at NC 2, {0, 0, 0, 1} at 4) at any address; the end compares the
// host's bytes too, at their DIMM addresses.
`timescale 1ns / 1ps
module tb_migpair #(parameter int CL = 29, parameter bit DM = 1'b0, parameter int DMB = 64,
                    parameter bit HM = 1'b0, parameter int NC = 2, parameter int BHW = 0,
                    parameter int BHB = 16, parameter int RCP = 0, parameter int WCP = 0,
                    parameter int WOC = 1, parameter bit RS = 1'b0, parameter int RD = 32);
  localparam int LNC = $clog2(NC);
  localparam int DA  = CL + LNC;               // the DIMM's byte address bits
  localparam int HB  = NC == 2 ? 2 : 0;        // the host's ID bit (otpu_mig_hmux HB)
  localparam logic [3:0] HID = 4'(1) << HB;
  logic clk = 1'b0, rst = 1'b1;
  always #1.667 clk = !clk;

  int unsigned seed, nmax, blocks, wpct, ppct, ot, gapw, lat, jit, rbw, st, wlat, maxc;
  int unsigned hn, hmaxb, hblk, hwpct, hppct, hot, hgapw, hst;
  bit seqrun, hi;
  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    if (!$value$plusargs("n=%d", nmax)) nmax = 2000;
    if (!$value$plusargs("blocks=%d", blocks)) blocks = 64;
    if (!$value$plusargs("wpct=%d", wpct)) wpct = 40;
    if (!$value$plusargs("ppct=%d", ppct)) ppct = 20;
    if (!$value$plusargs("ot=%d", ot)) ot = 16;
    if (!$value$plusargs("gapw=%d", gapw)) gapw = 10;
    if (!$value$plusargs("lat=%d", lat)) lat = 40;
    if (!$value$plusargs("jit=%d", jit)) jit = 20;
    if (!$value$plusargs("rbw=%d", rbw)) rbw = 100;
    if (!$value$plusargs("st=%d", st)) st = 10;
    if (!$value$plusargs("wlat=%d", wlat)) wlat = 20;
    if (!$value$plusargs("maxc=%d", maxc)) maxc = 2000000;
    if (!$value$plusargs("hn=%d", hn)) hn = HM ? 1000 : 0;
    if (!$value$plusargs("hmaxb=%d", hmaxb)) hmaxb = 64;
    if (!$value$plusargs("hblk=%d", hblk)) hblk = 16;
    if (!$value$plusargs("hwpct=%d", hwpct)) hwpct = 50;
    if (!$value$plusargs("hppct=%d", hppct)) hppct = 30;
    if (!$value$plusargs("hot=%d", hot)) hot = 8;
    if (!$value$plusargs("hgapw=%d", hgapw)) hgapw = 10;
    if (!$value$plusargs("hst=%d", hst)) hst = 10;
    if (!HM) hn = 0;
    seqrun = $test$plusargs("seq");
    hi = $test$plusargs("hi");
    if (seqrun) begin wpct = 0; jit = 0; st = 0; gapw = 0; end
    void'($urandom(seed));
  end

  // ---- the pair's ports
  logic [NC-1:0]        s_awvalid, s_awready, s_wvalid, s_wready, s_wlast, s_bvalid, s_bready;
  logic [NC-1:0]        s_arvalid, s_arready, s_rvalid, s_rready, s_rlast;
  logic [NC-1:0][33:0]  s_awaddr, s_araddr;
  logic [NC-1:0][3:0]   s_awlen, s_arlen;
  logic [NC-1:0][255:0] s_wdata, s_rdata;
  logic [NC-1:0][31:0]  s_wstrb;
  logic [1:0]           s_bresp;
  logic [NC-1:0][1:0]   s_rresp;
  logic              m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready;
  logic              m_arvalid, m_arready, m_rvalid, m_rready, m_rlast;
  logic [3:0]        m_awid, m_arid, m_bid, m_rid;
  logic [DA-1:0]     m_awaddr, m_araddr;
  logic [7:0]        m_awlen, m_arlen;
  logic [511:0]      m_wdata, m_rdata;
  logic [63:0]       m_wstrb;
  logic [1:0]        m_bresp, m_rresp;

  // HM: the pair's port into the hmux (p_*), the host's (h_*), the hmux's the DIMM's (m_*)
  logic              p_awvalid, p_awready, p_wvalid, p_wready, p_wlast, p_bvalid, p_bready;
  logic              p_arvalid, p_arready, p_rvalid, p_rready, p_rlast;
  logic [3:0]        p_awid, p_arid, p_bid, p_rid;
  logic [DA-1:0]     p_awaddr, p_araddr;
  logic [7:0]        p_awlen, p_arlen;
  logic [511:0]      p_wdata, p_rdata;
  logic [63:0]       p_wstrb;
  logic [1:0]        p_bresp, p_rresp;
  logic              h_awvalid, h_awready, h_wvalid, h_wready, h_wlast, h_bvalid, h_bready;
  logic              h_arvalid, h_arready, h_rvalid, h_rready, h_rlast;
  logic [DA-1:0]     h_awaddr, h_araddr;
  logic [7:0]        h_awlen, h_arlen;
  logic [511:0]      h_wdata, h_rdata;
  logic [63:0]       h_wstrb;
  logic [1:0]        h_bresp, h_rresp;

  if (HM) begin : g_hm
    otpu_mig_pair #(.IDW(4), .RD(RD), .CL(CL), .NC(NC), .BHW(BHW), .BHB(BHB), .WOC(WOC),
                    .RS(RS)) dut (.clk, .rst, .*,
      .m_awvalid(p_awvalid), .m_awready(p_awready), .m_awid(p_awid), .m_awaddr(p_awaddr),
      .m_awlen(p_awlen), .m_wvalid(p_wvalid), .m_wready(p_wready), .m_wdata(p_wdata),
      .m_wstrb(p_wstrb), .m_wlast(p_wlast), .m_bvalid(p_bvalid), .m_bready(p_bready),
      .m_bid(p_bid), .m_bresp(p_bresp), .m_arvalid(p_arvalid), .m_arready(p_arready),
      .m_arid(p_arid), .m_araddr(p_araddr), .m_arlen(p_arlen), .m_rvalid(p_rvalid),
      .m_rready(p_rready), .m_rid(p_rid), .m_rdata(p_rdata), .m_rresp(p_rresp),
      .m_rlast(p_rlast));
    otpu_mig_hmux #(.IDW(4), .AW(DA), .HB(HB), .RCP(RCP), .WCP(WCP)) u_hm (.clk, .rst, .*);
  end else begin : g_dir
    otpu_mig_pair #(.IDW(4), .RD(RD), .CL(CL), .NC(NC), .BHW(BHW), .BHB(BHB), .WOC(WOC),
                    .RS(RS)) dut (.clk, .rst, .*);
  end

  int unsigned cyc = 0;
  bit fail = 1'b0;
  always @(posedge clk) cyc <= cyc + 1;

  // ---- the DIMM's port
  int unsigned rbeats = 0, wbeats = 0, rfirst = 0, rlastc = 0;
  logic        dump = 1'b0;
  always @(posedge clk)
    if (!rst) begin
      if (m_rvalid && m_rready) begin
        rbeats++;
        if (rfirst == 0) rfirst = cyc;
        rlastc = cyc;
      end
      if (m_wvalid && m_wready) wbeats++;
    end

  if (!DM) begin : g_mod
    typedef struct packed { logic [3:0] id; logic [DA-1:0] a; logic [7:0] len; } cmd_t;
    logic [511:0] mem [longint unsigned];     // by 512-bit beat (byte address >> 6)
    cmd_t        arq[$], awq[$];
    int unsigned art[$];                      // the cycle each queued read may start
    logic [3:0]  bq[$];
    int unsigned bt[$];
    int unsigned rk = 0, wk = 0;              // the head read's / write's next beat

    function automatic bit idle();
      return awq.size() == 0 && arq.size() == 0 && bq.size() == 0;
    endfunction
    function automatic int errs();
      return 0;
    endfunction
    function automatic int nsparse();
      return 0;
    endfunction
    // the DIMM's byte d
    function automatic logic [7:0] mbyte(input longint unsigned d);
      return mem.exists(d >> 6) ? mem[d >> 6][8 * (d & 63) +: 8] : 8'h00;
    endfunction

    function automatic bit bad(input cmd_t c);
      return c.a[5:0] != 6'd0 || ((64'(c.a) >> 12) != ((64'(c.a) + (64'(c.len) << 6) + 63) >> 12)) ||
             (HM && c.id == HID ? 1'b0 :
              (c.a[12 +: LNC] != c.id[3 -: LNC] || (c.id & (4'hF >> LNC)) != 4'd0));
    endfunction

    always @(posedge clk) begin
      if (rst) begin
        m_arready <= 1'b0; m_awready <= 1'b0; m_wready <= 1'b0; m_rvalid <= 1'b0; m_bvalid <= 1'b0;
      end else begin
        if (m_arvalid && m_arready) begin
          arq.push_back({m_arid, m_araddr, m_arlen});
          art.push_back(cyc + lat + $urandom_range(jit));
          if (bad({m_arid, m_araddr, m_arlen})) begin
            $display("FAIL model: AR id %h addr %h len %0d", m_arid, m_araddr, m_arlen); fail = 1;
          end
        end
        if (m_awvalid && m_awready) begin
          awq.push_back({m_awid, m_awaddr, m_awlen});
          if (bad({m_awid, m_awaddr, m_awlen})) begin
            $display("FAIL model: AW id %h addr %h len %0d", m_awid, m_awaddr, m_awlen); fail = 1;
          end
        end
        if (m_wvalid && m_wready) begin
          automatic longint unsigned b = (64'(awq[0].a) >> 6) + 64'(wk);
          automatic logic [511:0] o = mem.exists(b) ? mem[b] : '0;
          for (int i = 0; i < 64; i++) if (m_wstrb[i]) o[8 * i +: 8] = m_wdata[8 * i +: 8];
          mem[b] = o;
          if (m_wlast != (wk == 32'(awq[0].len))) begin
            $display("FAIL model: WLAST %0d at beat %0d of %0d", m_wlast, wk, awq[0].len + 1);
            fail = 1;
          end
          if (m_wlast) begin
            bq.push_back(awq[0].id); bt.push_back(cyc + wlat); void'(awq.pop_front()); wk = 0;
          end else wk++;
        end
        if (m_bvalid && m_bready) begin void'(bq.pop_front()); void'(bt.pop_front()); end
        if (m_rvalid && m_rready) begin
          if (m_rlast) begin void'(arq.pop_front()); void'(art.pop_front()); rk = 0; end
          else rk++;
        end
        // the next cycle's outputs
        m_arready <= arq.size() < 64 && $urandom_range(99) >= st;
        m_awready <= awq.size() < 32 && $urandom_range(99) >= st;
        m_wready  <= awq.size() != 0 && $urandom_range(99) >= st;   // (W after its AW)
        if (!(m_rvalid && !m_rready)) begin
          if (arq.size() != 0 && cyc + 1 >= art[0] && $urandom_range(99) < rbw) begin
            automatic longint unsigned b = (64'(arq[0].a) >> 6) + 64'(rk);
            m_rvalid <= 1'b1;
            m_rid    <= arq[0].id;
            m_rdata  <= mem.exists(b) ? mem[b] : '0;
            m_rlast  <= rk == 32'(arq[0].len);
            m_rresp  <= 2'b00;
          end else m_rvalid <= 1'b0;
        end
        if (!(m_bvalid && !m_bready)) begin
          m_bvalid <= bq.size() != 0 && cyc + 1 >= bt[0];
          m_bid    <= bq.size() != 0 ? bq[0] : 4'd0;
          m_bresp  <= 2'b00;
        end
      end
    end
  end else begin : g_mod
    int     merr;
    longint mrd, mwr;
    otpu_dimm_model #(.BEATS(DMB), .DAW(DA), .IDW(4), .DIMM(0), .NC(NC), .BHW(BHW), .BHB(BHB)) u_dm (.clk, .rst, .dump,
      .awvalid(m_awvalid), .awready(m_awready), .awid(m_awid), .awaddr(m_awaddr), .awlen(m_awlen),
      .wvalid(m_wvalid), .wready(m_wready), .wdata(m_wdata), .wstrb(m_wstrb), .wlast(m_wlast),
      .bvalid(m_bvalid), .bready(m_bready), .bid(m_bid), .bresp(m_bresp),
      .arvalid(m_arvalid), .arready(m_arready), .arid(m_arid), .araddr(m_araddr), .arlen(m_arlen),
      .rvalid(m_rvalid), .rready(m_rready), .rid(m_rid), .rdata(m_rdata), .rresp(m_rresp),
      .rlast(m_rlast), .nrd(mrd), .nwr(mwr), .nerr(merr));

    function automatic bit idle();
      return 1'b1;
    endfunction
    function automatic int errs();
      return merr;
    endfunction
    function automatic int nsparse();             // the beats in the model's sparse store
      automatic int n = 0;
      for (int c = 0; c < NC; c++) n += g_mod.u_dm.hi[c].size();
      return n;
    endfunction
    // the DIMM's byte d: channel d[12 +: LNC]'s 32-byte word {d[DA-1:12+LNC], d[11:5]} (the
    // model's), dense or sparse
    function automatic logic [7:0] mbyte(input longint unsigned d);
      automatic int s = int'((d >> 12) & 64'(NC - 1));
      automatic longint w = longint'(((d >> (12 + LNC)) << 7) | ((d >> 5) & 127));
      automatic logic [255:0] v = w < longint'(DMB) ? g_mod.u_dm.mem[s][int'(w)] :
                                  g_mod.u_dm.hi[s].exists(w) ? g_mod.u_dm.hi[s][w] : '0;
      return v[8 * (d & 31) +: 8];
    endfunction
  end

  // ---- the masters
  logic [7:0] shadow [NC][longint unsigned];   // the channels' bytes

  // block k's channel byte address
  function automatic longint unsigned baddr(input int unsigned k);
    if (!hi) return 64'(k) * 512;
    return 64'(k >> 4) * 512 + (64'(k & 15) << (CL - 4));
  endfunction
  int unsigned done [NC], nrd [NC], nwr [NC], inflt [NC];
  bit pend [NC];

  for (genvar s = 0; s < NC; s++) begin : g_m
    typedef struct packed { logic we; logic [33:0] a; logic [7:0] len; } b_t;
    b_t  rq[$], wq[$];                      // reads / writes issued, not done
    b_t  p;                                 // the burst offered
    logic [255:0] wd[$];
    logic [31:0]  ws[$];
    logic         wl[$];
    int unsigned  rb = 0, seqb = 0;
    int unsigned  rinb [longint unsigned], winb [longint unsigned];   // per block: reads / writes

    function automatic logic [7:0] sh(input longint unsigned o);
      return shadow[s].exists(o) ? shadow[s][o] : 8'h00;
    endfunction
    function automatic void shw(input longint unsigned o, input logic [7:0] v);
      shadow[s][o] = v;
    endfunction

    always @(posedge clk) begin
      if (rst) begin
        done[s] = 0; nrd[s] = 0; nwr[s] = 0; pend[s] = 1'b0;
        s_arvalid[s] <= 1'b0; s_awvalid[s] <= 1'b0; s_wvalid[s] <= 1'b0;
      end else begin
        // the offered burst taken
        if (pend[s] && (p.we ? s_awvalid[s] && s_awready[s] : s_arvalid[s] && s_arready[s])) begin
          automatic longint unsigned bk = 64'(p.a) / 512;
          pend[s] = 1'b0;
          done[s]++;
          if (p.we) begin
            for (int k = 0; k < p.len; k++) begin
              automatic logic [255:0] d;
              automatic logic [31:0]  m;
              for (int i = 0; i < 8; i++) d[32 * i +: 32] = $urandom;
              m = $urandom_range(99) < ppct ? $urandom : 32'hFFFF_FFFF;
              for (int i = 0; i < 32; i++) if (m[i]) shw(64'(p.a) + 64'(32 * k + i), d[8 * i +: 8]);
              wd.push_back(d); ws.push_back(m); wl.push_back(k == p.len - 1);
            end
            wq.push_back(p);
            winb[bk] = winb.exists(bk) ? winb[bk] + 1 : 1;
            nwr[s]++;
          end else begin
            rq.push_back(p);
            rinb[bk] = rinb.exists(bk) ? rinb[bk] + 1 : 1;
            nrd[s]++;
          end
        end
        if (s_wvalid[s] && s_wready[s]) begin
          void'(wd.pop_front()); void'(ws.pop_front()); void'(wl.pop_front());
        end
        if (s_bvalid[s]) begin
          automatic longint unsigned bk = 64'(wq[0].a) / 512;
          winb[bk]--;
          void'(wq.pop_front());
        end
        if (s_rvalid[s]) begin
          for (int i = 0; i < 32; i++)
            if (s_rdata[s][8 * i +: 8] !== sh(64'(rq[0].a) + 64'(32 * rb + i)) && !fail) begin
              $display("FAIL bridge %0d: read %h beat %0d byte %0d: %h, expected %h", s, rq[0].a,
                       rb, i, s_rdata[s][8 * i +: 8], sh(64'(rq[0].a) + 64'(32 * rb + i)));
              fail = 1;
            end
          if (s_rlast[s] != (rb == 32'(rq[0].len) - 1)) begin
            $display("FAIL bridge %0d: RLAST %0d at beat %0d of %0d", s, s_rlast[s], rb,
                     rq[0].len);
            fail = 1;
          end
          if (s_rlast[s]) begin
            automatic longint unsigned bk = 64'(rq[0].a) / 512;
            rinb[bk]--;
            void'(rq.pop_front());
            rb = 0;
          end else rb++;
        end
        // a new burst to offer
        if (!pend[s] && done[s] < nmax && rq.size() + wq.size() < ot) begin
          automatic int unsigned blk = seqrun ? seqb % blocks : $urandom_range(blocks - 1);
          automatic int unsigned nb  = seqrun ? 4 : 1 + $urandom_range(3);   // channel beats
          automatic int unsigned b0  = seqrun ? 0 : $urandom_range(4 - nb);  // in the block
          automatic logic we = !seqrun && $urandom_range(99) < wpct;
          automatic longint unsigned ba = baddr(blk), bk = ba / 512;
          if (!(we ? rinb.exists(bk) && rinb[bk] != 0 : winb.exists(bk) && winb[bk] != 0)) begin
            p = '{we: we, a: 34'(ba + 64'(b0 * 128)), len: 8'(nb * 4)};
            pend[s] = 1'b1;
            if (seqrun) seqb++;
          end
        end
        inflt[s] = rq.size() + wq.size();
        s_arvalid[s] <= pend[s] && !p.we;
        s_awvalid[s] <= pend[s] && p.we;
        s_araddr[s]  <= 34'(p.a);
        s_awaddr[s]  <= 34'(p.a);
        s_arlen[s]   <= 4'(p.len - 1);
        s_awlen[s]   <= 4'(p.len - 1);
        if (!(s_wvalid[s] && !s_wready[s])) begin
          // W beats of the bursts already granted (wd holds only those), with gaps
          s_wvalid[s] <= wd.size() != 0 && $urandom_range(99) >= gapw;
          s_wdata[s]  <= wd.size() != 0 ? wd[0] : '0;
          s_wstrb[s]  <= ws.size() != 0 ? ws[0] : '0;
          s_wlast[s]  <= wl.size() != 0 ? wl[0] : 1'b0;
        end
      end
    end
    assign s_bready[s] = 1'b1;
    assign s_rready[s] = 1'b1;
  end

  // ---- the host (HM)
  logic [7:0] hshadow [longint unsigned];   // the host's bytes, by DIMM address
  int unsigned hdone = 0, hnrd = 0, hnwr = 0, hinflt = 0;
  bit hpend = 1'b0;

  if (HM) begin : g_h
    typedef struct packed { logic we; logic [DA-1:0] a; logic [7:0] n; } h_t;   // n: beats
    h_t  rq[$], wq[$];                      // reads / writes offered, not done
    h_t  p;                                 // the burst offered
    logic [511:0] wd[$];
    logic [63:0]  ws[$];
    logic         wl[$];
    int unsigned  rb = 0;
    int unsigned  rinb [longint unsigned], winb [longint unsigned];   // per 4 KiB block

    function automatic logic [7:0] sh(input longint unsigned a);
      return hshadow.exists(a) ? hshadow[a] : 8'h00;
    endfunction

    always @(posedge clk) begin
      if (rst) begin
        hdone = 0; hnrd = 0; hnwr = 0; hpend = 1'b0;
        h_arvalid <= 1'b0; h_awvalid <= 1'b0; h_wvalid <= 1'b0; h_rready <= 1'b0;
        h_bready <= 1'b0;
      end else begin
        if (hpend && (p.we ? h_awvalid && h_awready : h_arvalid && h_arready)) begin
          hpend = 1'b0;
          hdone++;
        end
        if (h_wvalid && h_wready) begin
          void'(wd.pop_front()); void'(ws.pop_front()); void'(wl.pop_front());
        end
        if ((h_bvalid && wq.size() == 0) || (h_rvalid && rq.size() == 0)) begin
          $display("FAIL host: B %0d / R %0d with %0d writes, %0d reads in flight", h_bvalid,
                   h_rvalid, wq.size(), rq.size());
          fail = 1;
        end
        if (h_bvalid && h_bready && wq.size() != 0) begin
          automatic longint unsigned bk = 64'(wq[0].a) >> 12;
          winb[bk]--;
          void'(wq.pop_front());
        end
        if (h_rvalid && h_rready && rq.size() != 0) begin
          for (int i = 0; i < 64; i++)
            if (h_rdata[8 * i +: 8] !== sh(64'(rq[0].a) + 64'(64 * rb + i)) && !fail) begin
              $display("FAIL host: read %h beat %0d byte %0d: %h, expected %h", rq[0].a, rb, i,
                       h_rdata[8 * i +: 8], sh(64'(rq[0].a) + 64'(64 * rb + i)));
              fail = 1;
            end
          if (h_rlast != (rb == 32'(rq[0].n) - 1)) begin
            $display("FAIL host: RLAST %0d at beat %0d of %0d", h_rlast, rb, rq[0].n);
            fail = 1;
          end
          if (h_rlast) begin
            automatic longint unsigned bk = 64'(rq[0].a) >> 12;
            rinb[bk]--;
            void'(rq.pop_front());
            rb = 0;
          end else rb++;
        end
        // a new burst: n beats from beat b0 of block blk of the DIMM's top hblk
        if (!hpend && hdone < hn && rq.size() + wq.size() < hot) begin
          automatic int unsigned blk = $urandom_range(hblk - 1);
          automatic int unsigned n   = 1 + $urandom_range(hmaxb - 1);
          automatic int unsigned b0  = $urandom_range(64 - n);
          automatic logic we = $urandom_range(99) < hwpct;
          automatic longint unsigned bk = (64'd1 << (DA - 12)) - 64'(hblk) + 64'(blk);
          if (!(we ? rinb.exists(bk) && rinb[bk] != 0 : winb.exists(bk) && winb[bk] != 0)) begin
            p = '{we: we, a: DA'((bk << 12) + 64'(b0 * 64)), n: 8'(n)};
            hpend = 1'b1;
            if (we) begin
              for (int k = 0; k < n; k++) begin
                automatic logic [511:0] d;
                automatic logic [63:0]  m;
                for (int i = 0; i < 16; i++) d[32 * i +: 32] = $urandom;
                m = $urandom_range(99) < hppct ? {$urandom, $urandom} : '1;
                for (int i = 0; i < 64; i++)
                  if (m[i]) hshadow[64'(p.a) + 64'(64 * k + i)] = d[8 * i +: 8];
                wd.push_back(d); ws.push_back(m); wl.push_back(k == n - 1);
              end
              wq.push_back(p);
              winb[bk] = winb.exists(bk) ? winb[bk] + 1 : 1;
              hnwr++;
            end else begin
              rq.push_back(p);
              rinb[bk] = rinb.exists(bk) ? rinb[bk] + 1 : 1;
              hnrd++;
            end
          end
        end
        hinflt = rq.size() + wq.size();
        h_arvalid <= hpend && !p.we;
        h_awvalid <= hpend && p.we;
        h_araddr  <= p.a;
        h_awaddr  <= p.a;
        h_arlen   <= p.n - 8'd1;
        h_awlen   <= p.n - 8'd1;
        if (!(h_wvalid && !h_wready)) begin
          // W beats of the bursts offered (they may lead their AW), with gaps
          h_wvalid <= wd.size() != 0 && $urandom_range(99) >= hgapw;
          h_wdata  <= wd.size() != 0 ? wd[0] : '0;
          h_wstrb  <= ws.size() != 0 ? ws[0] : '0;
          h_wlast  <= wl.size() != 0 ? wl[0] : 1'b0;
        end
        h_rready <= $urandom_range(99) >= hst;
        h_bready <= $urandom_range(99) >= hst;
      end
    end
  end

  // ---- the end
  initial begin
    repeat (20) @(posedge clk);
    rst <= 1'b0;
  end
  int unsigned fin = 0;                     // the end: 1 the dump asked, 2 seen
  function automatic bit masters_done();
    for (int c = 0; c < NC; c++) if (done[c] < nmax || inflt[c] != 0 || pend[c]) return 1'b0;
    return 1'b1;
  endfunction
  function automatic string per(input int unsigned v [NC]);
    automatic string t = "";
    for (int c = 0; c < NC; c++) t = {t, $sformatf(c ? " %0d" : "%0d", v[c])};
    return t;
  endfunction
  always @(posedge clk) begin
    if (fail) begin
      $display("FAIL cycles %0d", cyc);
      $finish;
    end
    if (cyc > maxc) begin
      $display("TIMEOUT cycles %0d done %s in flight %s host %0d %0d", cyc, per(done), per(inflt),
               hdone, hinflt);
      $finish;
    end
    if (!rst && masters_done() && hdone >= hn && hinflt == 0 && !hpend && g_mod.idle() &&
        fin < 2) begin
      dump <= fin == 0;
      fin++;
    end else if (fin == 2) begin
      if (g_mod.errs() != 0) begin
        $display("FAIL model: %0d errors", g_mod.errs()); fail = 1;
      end
      for (int c = 0; c < NC; c++) foreach (shadow[c][o]) if (!fail) check(c, o, shadow[c][o]);
      foreach (hshadow[a]) if (!fail) hcheck(a, hshadow[a]);
      if (DM) $display("model: %0d beats in the sparse store", g_mod.nsparse());
      if (HM) $display("host: reads %0d writes %0d", hnrd, hnwr);
      if (fail) $display("FAIL cycles %0d", cyc);
      else
        $display("PASS cycles %0d rbeats %0d wbeats %0d reads %s writes %s rrate %0.3f",
                 cyc, rbeats, wbeats, per(nrd), per(nwr), real'(rbeats) / real'(rlastc - rfirst + 1));
      $finish;
    end
  end
  // the bank-half window's swap of a DIMM byte address (its own inverse: otpu_mig_pair's dimm)
  function automatic longint unsigned bhw(input longint unsigned d);
    longint unsigned e = d;
    if (BHW != 0 && (d >> (BHW + LNC)) == 0) begin
      e[BHB] = d[BHW - 1 + LNC];
      e[BHW - 1 + LNC] = d[BHB];
    end
    return e;
  endfunction
  // ($finish ends the run at the time step's end: a map failure only marks it; DM's mbyte takes
  // the address before the swap, the other model's after it)
  task automatic check(input int s, input longint unsigned o, input logic [7:0] e);
    automatic longint unsigned d = (o >> 12 << (12 + LNC)) | (64'(s) << 12) | (o & 64'hFFF);
    automatic logic [7:0] v = g_mod.mbyte(DM ? d : bhw(d));
    if (v !== e) begin
      $display("FAIL map: channel %0d byte %h (DIMM %h): %h, expected %h", s, o, d, v, e);
      fail = 1;
    end
  endtask
  task automatic hcheck(input longint unsigned d, input logic [7:0] e);
    automatic logic [7:0] v = g_mod.mbyte(DM ? bhw(d) : d);
    if (v !== e) begin
      $display("FAIL host: DIMM byte %h: %h, expected %h", d, v, e);
      fail = 1;
    end
  endtask
endmodule
