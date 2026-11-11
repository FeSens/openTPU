// otpu_native_dram alone (tests/test_native_dram.py): random traffic on its ports A, SW, B and W
// (W where P = NCH x CB > D) through the native memory model (otpu_native_mem: random stalls,
// latency and jitter, +axi_* / +nat_* as there), against a reference memory (view) that takes
// every write as it is issued.
//
// The traffic runs in epochs. An epoch gives each 4 KiB region one owner: read-only, B, A or SW.
// Every port reads what it may: B its own regions and the read-only ones, A likewise, W (whole
// wide chunks) the read-only ones; B, A and SW write their own (B word-masked chunks, A and SW
// byte-enabled words; SW also in strides and runs, as QST's streams). So a read's data is the
// view at its issue: within a port requests go out in order, and no other port writes what it
// reads. Between epochs the testbench waits for the adapter's rd_idle and wr_idle with no request
// pending, and the next epoch's owners read what the last epoch's wrote and write what it read:
// a write that wr_idle let pass before it was in memory, or a read that rd_idle let pass before
// its data, returns the wrong data. Every cycle rd_idle must mean no read is outstanding (the
// data returning this cycle consumed), and at each epoch's end the model's memory must be the
// view. Each port's reads return in order with their tags (B) and are checked against the view
// as issued; port W's beats per lane (channel c's: logical beat c ^ h of its read, h the
// read's channel hash), each lane in its reads' order.
//   +seed=N +epochs=N +ops=N (requests per epoch and port, about) +busy=P (percent of cycles a
//   port with work has a request) +wcred=N (port W: at most N reads not yet in every lane, as
//   the MXU's chunk FIFO entries; 0: no bound)
// +wstream=N instead: N port W reads back to back over the read-only memory, no other traffic
// (TBN WSTREAM: the cycles, the reads per cycle against the port's peak of one, and the lanes'
// largest skew: reads in some lane and not yet in all); +bstream=N N port B reads so, alone or
// beside the W stream (TBN BSTREAM: its cycles, and the W reads taken meanwhile: B goes before W
// on every channel); +bseq=1: each B read the next chunk (consecutive chunks on different channels,
// the hash's), +bseq=2: the next chunk on channel 0 (one channel back to back). WPOST: the write data posted at the take (otpu_native_dram WPOST, the model's
// write-data FIFO of 32 beats with n_wfree, as the WG targets' bridges); port A then reads only, as the
// slice's. XPOST: port W's beats posted by the model (otpu_native_dram XPOST): each W read
// carries an entry (its index mod 2^9), and every lane's posted beat is checked (entry, data, in
// its lane's order); rd_idle must cover the posted beats (the adapter's counts of them). A third
// of the W reads are then one beat (w_one: a D-byte chunk, posted on its channel's lane only).
// XSTG: XSTG register stages each way between the traffic (the slice's side) and the adapter
// (otpu_xstg, as otpu_board NAT_XSTG): its queues and idles, the posts and their counts.
// AB: port A's read bytes (32, or CB: its beat, the WG targets' scales). RHB / WHB: the native port's read
// and posted write data in half beats (otpu_native_dram, otpu_native_mem RHB / WHB).
// +astream=N (with the streams): N port A reads back to back, each the next AB-byte block (AB = CB:
// the next beat, so on the next channel: the MXU's scale stream at its peak), beside the W and B
// streams if asked (TBN ASTREAM: its cycles, its reads a cycle).
module tb_ndram;
  parameter int NCH   = 2;
  parameter int CB    = 64;
  parameter int D     = 128;
  parameter int WORDS = 1 << 16;          // the memory, logical words
  parameter int RD    = 256 / NCH;        // the adapter's depths (otpu_native_dram's defaults)
  parameter int AD    = 64;
  parameter int XD    = 64;
  parameter int WQD   = 32;
  parameter int APF   = 32;
  parameter bit SW_FILL = 1'b1;
  parameter bit WPOST = 1'b0;
  parameter bit SWG   = 1'b0;
  parameter bit XPOST = 1'b0;
  parameter bit RB_BLOCK = 1'b0;        // the B FIFO in block RAM (otpu_native_dram RB_BLOCK)
  parameter int XSTG  = 0;
  parameter int XSTG_P = XSTG;
  parameter int AB    = 32;               // port A's read bytes (otpu_native_dram AB)
  parameter bit RHB   = 1'b0;             // half beats (otpu_native_dram, otpu_native_mem)
  parameter bit WHB   = 1'b0;
  localparam int P = NCH * CB;
  localparam int CL = $clog2(NCH);
  localparam int PWW = $clog2(P / 4);
  localparam int CAW = 30 - PWW;
  localparam int DW = D / 4;              // words a chunk
  localparam int PWD = P / 4;             // words a wide chunk
  localparam bit HAS_W = P > D;
  localparam int RGW = 1024;              // words a region
  localparam int NR = WORDS / RGW;
  localparam int O_RO = 0, O_B = 1, O_A = 2, O_SW = 3;

  logic clk = 1'b0, rst = 1'b1;
  always #5 clk = ~clk;

  logic a_rdy, a_req = 0, a_we = 0, a_rvalid;
  logic [31:0] a_addr = 0, a_wdata = 0;
  logic [3:0] a_be = 0;
  logic [AB*8-1:0] a_rdata;
  logic sw_rdy, sw_req = 0;
  logic [31:0] sw_addr = 0, sw_wdata = 0;
  logic [3:0] sw_be = 0;
  logic b_rdy, b_req = 0, b_tag = 0, b_we = 0, b_rvalid, b_rtag;
  logic [DW-1:0] b_wmask = 0;
  logic [D*8-1:0] b_wdata = 0, b_rdata;
  logic [31:0] b_addr = 0;
  logic [CL-1:0] b_par;
  logic w_rdy, w_req = 0;
  logic [31:0] w_addr = 0;
  logic [NCH-1:0] w_bv;
  logic [P*8-1:0] w_bd;
  logic [8:0] w_slot = 0;
  logic w_one = 0;
  logic [NCH-1:0] pwe;                    // XPOST: the posted beats, their counts
  logic [NCH-1:0][8:0] pslot, ncslot;
  logic [NCH-1:0][CB*8-1:0] pd;
  logic [NCH-1:0][10:0] pcg, pcnt;
  logic [NCH-1:0][1:0] ncsrc;
  logic wr_idle, rd_idle;
  logic [NCH-1:0] ncv, ncr, ncwe, nwv, nwr, nrv;
  logic [NCH-1:0][CAW-1:0] nca;
  logic [NCH-1:0][CB*8-1:0] nwd, nrd;
  logic [NCH-1:0][CB-1:0] nwm;
  logic [NCH-1:0][15:0] nwdone;
  logic [NCH-1:0][5:0] nwfree;

  // the fold the slice computes for port B (otpu_slice chash)
  function automatic logic [CL-1:0] fold(input logic [31:0] w);
    logic [CL-1:0] h;
    h = '0;
    for (int i = PWW; i < 32; i++) h[(i - PWW) % CL] ^= w[i];
    return h;
  endfunction
  assign b_par = fold(b_addr);
  function automatic logic [CL-1:0] chan(input logic [31:0] w);   // the word's channel
    return CL'((w % PWD) / (CB / 4)) ^ fold(w);
  endfunction

  // the adapter's side of u_xs (the traffic's signals at XSTG 0), the model's posts and counts
  logic d_rst, da_rdy, da_req, da_we, da_rvalid, dsw_rdy, dsw_req, db_rdy, db_req, db_tag, db_we;
  logic db_rvalid, db_rtag, dw_rdy, dw_req, dw_one, d_wr_idle, d_rd_idle;
  logic [31:0] da_addr, da_wdata, dsw_addr, dsw_wdata, db_addr, dw_addr;
  logic [3:0] da_be, dsw_be;
  logic [AB*8-1:0] da_rdata;
  logic [DW-1:0] db_wmask;
  logic [D*8-1:0] db_wdata, db_rdata;
  logic [CL-1:0] db_par;
  logic [8:0] dw_slot;
  logic [NCH-1:0] dw_bv, mwe;
  logic [P*8-1:0] dw_bd;
  logic [NCH-1:0][8:0] mslot;
  logic [NCH-1:0][CB*8-1:0] mpd;
  logic [NCH-1:0][10:0] mpcg, dpcnt, dpcg;
  logic [NCH-1:0][4:0][8:0] xslot, xslot_s;
  logic [NCH-1:0][D*8-1:0] xpd, xpd_s;
  for (genvar c = 0; c < NCH; c++) begin : g_pc
    assign xslot[c] = {5{mslot[c]}};
    assign xpd[c] = (D*8)'(mpd[c]);
    assign pslot[c] = xslot_s[c][0];
    assign pd[c] = (CB*8)'(xpd_s[c]);
  end
  assign pcg = dpcg;
  otpu_xstg #(.N(XSTG), .NP(XSTG_P), .D(D), .NCH(NCH), .CB(CB), .AB(AB), .XSW(9), .XQB(0)) u_xs (
    .clk, .p_clk(clk), .s_rst(rst), .s_flush(3'b0),
    .s_a_rdy(a_rdy), .s_a_req(a_req), .s_a_we(a_we), .s_a_addr(a_addr), .s_a_wdata(a_wdata),
    .s_a_be(a_be), .s_a_rvalid(a_rvalid), .s_a_rdata(a_rdata),
    .s_sw_rdy(sw_rdy), .s_sw_req(sw_req), .s_sw_addr(sw_addr), .s_sw_wdata(sw_wdata),
    .s_sw_be(sw_be),
    .s_b_rdy(b_rdy), .s_b_req(b_req), .s_b_tag(b_tag), .s_b_we(b_we), .s_b_wmask(b_wmask),
    .s_b_wdata(b_wdata), .s_b_addr(b_addr), .s_b_par(b_par), .s_b_rvalid(b_rvalid),
    .s_b_rtag(b_rtag), .s_b_rdata(b_rdata),
    .s_w_rdy(w_rdy), .s_w_req(w_req), .s_w_addr(w_addr), .s_w_slot(w_slot), .s_w_one(w_one),
    .s_w_bv(w_bv), .s_w_bd(w_bd),
    .s_wr_idle(wr_idle), .s_rd_idle(rd_idle), .s_p_cnt(pcnt),
    .s_p_we(pwe), .s_p_q(), .s_p_slot(xslot_s), .s_p_d(xpd_s),
    .m_rst(d_rst), .m_flush(),
    .m_a_rdy(da_rdy), .m_a_req(da_req), .m_a_we(da_we), .m_a_addr(da_addr),
    .m_a_wdata(da_wdata), .m_a_be(da_be), .m_a_rvalid(da_rvalid), .m_a_rdata(da_rdata),
    .m_sw_rdy(dsw_rdy), .m_sw_req(dsw_req), .m_sw_addr(dsw_addr), .m_sw_wdata(dsw_wdata),
    .m_sw_be(dsw_be),
    .m_b_rdy(db_rdy), .m_b_req(db_req), .m_b_tag(db_tag), .m_b_we(db_we), .m_b_wmask(db_wmask),
    .m_b_wdata(db_wdata), .m_b_addr(db_addr), .m_b_par(db_par), .m_b_rvalid(db_rvalid),
    .m_b_rtag(db_rtag), .m_b_rdata(db_rdata),
    .m_w_rdy(dw_rdy), .m_w_req(dw_req), .m_w_addr(dw_addr), .m_w_slot(dw_slot),
    .m_w_one(dw_one), .m_w_bv(dw_bv), .m_w_bd(dw_bd),
    .m_wr_idle(d_wr_idle), .m_rd_idle(d_rd_idle), .m_p_cnt(dpcnt), .m_p_cntg(dpcg),
    .c_p_we(mwe), .c_p_q('0), .c_p_slot(xslot), .c_p_d(xpd), .c_p_cntg(mpcg));
  otpu_native_dram #(.D(D), .NCH(NCH), .CB(CB), .RD(RD), .AD(AD), .XD(XD), .WQD(WQD), .APF(APF),
                     .SW_FILL(SW_FILL), .WPOST(WPOST), .SWG(SWG), .XPOST(XPOST), .RB_BLOCK(RB_BLOCK),
                     .AB(AB), .RHB(RHB), .WHB(WHB)) dut (
    .clk, .rst(d_rst), .a_flush(1'b0),
    .a_rdy_x(da_rdy), .a_req_x(da_req), .a_we_x(da_we), .a_addr_x(da_addr),
    .a_wdata_x(da_wdata), .a_be_x(da_be), .a_rvalid(da_rvalid), .a_rdata(da_rdata),
    .sw_rdy(dsw_rdy), .sw_req(dsw_req), .sw_addr(dsw_addr), .sw_wdata(dsw_wdata),
    .sw_be(dsw_be),
    .b_rdy(db_rdy), .b_req(db_req), .b_tag(db_tag), .b_we(db_we), .b_wmask(db_wmask),
    .b_wdata(db_wdata), .b_addr(db_addr), .b_par(db_par), .b_rvalid(db_rvalid),
    .b_rtag(db_rtag), .b_rdata(db_rdata), .w_rdy(dw_rdy), .w_req(dw_req), .w_addr(dw_addr),
    .w_slot(dw_slot), .w_one(dw_one), .w_bv(dw_bv), .w_bd(dw_bd),
    .wr_idle(d_wr_idle), .rd_idle(d_rd_idle),
    .n_cvalid(ncv), .n_cready(ncr), .n_cwe(ncwe), .n_caddr(nca), .n_csrc(ncsrc),
    .n_cslot(ncslot), .n_wvalid(nwv),
    .n_wready(nwr), .n_wdata(nwd), .n_wmask(nwm), .n_rvalid(nrv), .n_rdata(nrd),
    .n_wdone(nwdone), .n_wfree(nwfree), .p_cntg(dpcg), .p_cnt(dpcnt));
  // twice the traffic's memory: A's runs (APF beats a channel) read on past the last region
  otpu_native_mem #(.WORDS(2 * WORDS), .LAT(20), .NCH(NCH), .CB(CB), .WFD(WPOST ? 32 : 0),
                    .XPOST(XPOST), .RHB(RHB), .WHB(WHB)) u_mem (
    .clk, .rst, .run_rst(1'b0),
    .n_cvalid(ncv), .n_cready(ncr), .n_cwe(ncwe), .n_caddr(nca), .n_csrc(ncsrc),
    .n_cslot(ncslot), .n_wvalid(nwv),
    .n_wready(nwr), .n_wdata(nwd), .n_wmask(nwm), .n_rvalid(nrv), .n_rdata(nrd),
    .n_wdone(nwdone), .n_wfree(nwfree), .p_we(mwe), .p_slot(mslot), .p_d(mpd), .p_cntg(mpcg),
    .dump(1'b0));

  // ---- the reference
  logic [31:0] view [WORDS];
  int owner [NR];
  typedef struct { logic tag; logic [D*8-1:0] d; logic [31:0] a; } eb_t;
  typedef struct { logic [CB*8-1:0] d; logic [31:0] a; logic [8:0] s; } ew_t;  // a lane's beat
  typedef struct { logic [AB*8-1:0] d; logic [31:0] a; } ea_t;
  eb_t eb [$];
  ew_t ew [NCH][$];                       // per lane
  longint wl_n [NCH];                     // per lane: W beats returned
  longint w_all = 0, w_skew = 0;          // W reads in every lane; the largest lead of a lane
  longint n_one = 0;                      // XPOST: one-beat W reads
  ea_t ea [$];
  longint cyc = 0, n_brd = 0, n_bwr = 0, n_ard = 0, n_awr = 0, n_sw = 0, n_wrd = 0;
  int seed = 1, epochs = 12, ops = 400, busy = 60, wstream = 0, bstream = 0, wcred = 0, astream = 0;
  int bseq = 0;
  bit in_stream = 1'b0;                   // the streams' phase (after the memory's fill)
  bit issuing = 1'b0;                     // the epoch's requests are going out
  int left_b, left_a, left_sw, left_w;    // requests left this epoch, per port
  int sw_mode = 0, sw_left = 0;           // SW: a run (1) or a stride (2) in progress
  logic [31:0] sw_next = 0, a_next = 0, b_next = 0;

  function automatic int rnd(input int n);
    return int'($urandom % n);
  endfunction
  function automatic int w_left();          // W beats outstanding, all lanes
    int n;
    n = 0;
    for (int c = 0; c < NCH; c++) n += ew[c].size();
    return n;
  endfunction
  // a random region with one of two owners (or -1)
  function automatic int region(input int o1, input int o2);
    int r;
    for (int k = 0; k < 64; k++) begin
      r = rnd(NR);
      if (owner[r] == o1 || owner[r] == o2) return r;
    end
    return -1;
  endfunction

  // the next request of each port, on the falling edge (held while not taken: *_took, the last
  // rising edge's)
  bit b_took = 1'b0, a_took = 1'b0, sw_took = 1'b0, w_took = 1'b0;
  always @(negedge clk) if (!rst) begin
    int r;
    // B
    if (!b_req || b_took) begin
      b_req = 1'b0;
      r = region(O_RO, O_B);
      if (issuing && left_b > 0 && (bstream != 0 || rnd(100) < busy) && r >= 0) begin
        b_req = 1'b1; left_b--;
        b_addr = 32'(r * RGW + rnd(RGW / DW) * DW);
        if (in_stream && bseq != 0) begin    // the stream's next chunk (bseq 2: on channel 0)
          b_addr = b_next;
          do b_next = (b_next + DW) % WORDS; while (bseq == 2 && chan(b_next) != 0);
        end
        b_tag = rnd(2);
        b_we = owner[r] == O_B && rnd(2) == 0;
        if (b_we) begin
          do b_wmask = DW'({$urandom, $urandom}); while (b_wmask == '0);
          if (rnd(2) == 0) b_wmask = '1;
          for (int k = 0; k < D / 4; k++) b_wdata[32 * k +: 32] = $urandom;
        end
      end
    end
    // A: word reads, a scale stream's (the next word) or anywhere; writes in its own regions
    if (!a_req || a_took) begin
      a_req = 1'b0;
      if (issuing && left_a > 0 && astream != 0) begin     // the scale stream: the next block
        a_req = 1'b1; a_we = 1'b0; a_addr = a_next; left_a--;
        a_next = (a_next + 32'(AB / 4)) % 32'(WORDS);
      end else if (issuing && left_a > 0 && rnd(100) < busy) begin
        r = int'(a_next) / RGW;
        if (rnd(4) != 0 && a_next < WORDS && (owner[r] == O_RO || owner[r] == O_A)) begin
          a_req = 1'b1; a_we = 1'b0; a_addr = a_next;
          r = -1;
        end else r = region(O_RO, O_A);
        if (r >= 0) begin
          a_req = 1'b1;
          a_addr = 32'(r * RGW + rnd(RGW));
          a_we = owner[r] == O_A && rnd(3) == 0 && !WPOST;
          a_wdata = $urandom;
          do a_be = 4'($urandom); while (a_be == 0);
        end
        if (a_req) begin left_a--; a_next = a_addr + 1; end
      end
    end
    // SW: byte-enabled words in its regions, alone, in runs (the gather merges them) or at a
    // stride (one word a beat: fill reads)
    if (!sw_req || sw_took) begin
      sw_req = 1'b0;
      if (issuing && left_sw > 0 && rnd(100) < busy) begin
        if (sw_left > 0 && int'(sw_next) < WORDS && owner[int'(sw_next) / RGW] == O_SW) begin
          sw_addr = sw_next; sw_left--;
          sw_next = sw_next + (sw_mode == 1 ? 1 : 32'(CB / 4 * (1 + rnd(2 * NCH))));
          sw_req = 1'b1;
        end else r = region(O_SW, O_SW);
        if (!sw_req && r >= 0) begin
          sw_addr = 32'(r * RGW + rnd(RGW));
          sw_mode = rnd(3); sw_left = sw_mode == 0 ? 0 : 4 + rnd(40);
          sw_next = sw_addr + 1;
          sw_req = 1'b1;
        end
        if (sw_req) begin
          left_sw--;
          sw_wdata = $urandom;
          sw_be = rnd(2) == 0 ? 4'hF : 4'($urandom);
          if (sw_be == 0) sw_be = 4'h1;
        end
      end
    end
    // W: wide chunks of the read-only regions
    if (HAS_W && (!w_req || w_took)) begin
      w_req = 1'b0;
      r = region(O_RO, O_RO);
      if (issuing && left_w > 0 && (wstream != 0 || rnd(100) < busy) && r >= 0 &&
          (wcred == 0 || n_wrd - w_all < wcred)) begin
        w_req = 1'b1; left_w--;
        w_one = XPOST && rnd(3) == 0;
        w_addr = w_one ? 32'(r * RGW + rnd(RGW / DW) * DW) : 32'(r * RGW + rnd(RGW / PWD) * PWD);
        w_slot = 9'(n_wrd);
      end
    end
  end

  // the rising edge: read data checked (in order), rd_idle, then the requests taken
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (!rst) begin
      if (b_rvalid) begin
        if (eb.size() == 0) $fatal(1, "tb_ndram: B data with no B read outstanding");
        if (b_rtag != eb[0].tag || b_rdata != eb[0].d)
          $fatal(1, "tb_ndram: B read of word %h: tag %0d data differs (expected tag %0d)",
                 eb[0].a, b_rtag, eb[0].tag);
        void'(eb.pop_front());
      end
      begin
        longint lo, hi;
        if (XPOST && w_bv != '0) $fatal(1, "tb_ndram: XPOST: W data through the adapter");
        for (int c = 0; c < NCH; c++)
          if (XPOST ? pwe[c] : w_bv[c]) begin
            if (ew[c].size() == 0) $fatal(1, "tb_ndram: W data on lane %0d with no W read outstanding", c);
            if ((XPOST ? pd[c] : w_bd[CB * 8 * c +: CB * 8]) != ew[c][0].d)
              $fatal(1, "tb_ndram: W read of word %h: lane %0d's beat differs", ew[c][0].a, c);
            if (XPOST && pslot[c] != ew[c][0].s)
              $fatal(1, "tb_ndram: W read of word %h: lane %0d's beat into entry %0d, not %0d",
                     ew[c][0].a, c, pslot[c], ew[c][0].s);
            void'(ew[c].pop_front());
            wl_n[c]++;
          end
        lo = wl_n[0]; hi = wl_n[0];
        for (int c = 1; c < NCH; c++) begin
          if (wl_n[c] < lo) lo = wl_n[c];
          if (wl_n[c] > hi) hi = wl_n[c];
        end
        w_all = lo;
        if (hi - lo > w_skew) w_skew = hi - lo;
      end
      if (a_rvalid) begin
        if (ea.size() == 0) $fatal(1, "tb_ndram: A data with no A read outstanding");
        if (a_rdata != ea[0].d) $fatal(1, "tb_ndram: A read of word %h differs", ea[0].a);
        void'(ea.pop_front());
      end
      if (rd_idle && (eb.size() != 0 || w_left() != 0 || ea.size() != 0))
        $fatal(1, "tb_ndram: rd_idle with reads outstanding (B %0d, W beats %0d, A %0d)",
               eb.size(), w_left(), ea.size());
      b_took = b_req && b_rdy; a_took = a_req && a_rdy; sw_took = sw_req && sw_rdy;
      w_took = w_req && w_rdy;
      if (b_req && b_rdy) begin
        if (b_we) begin
          for (int k = 0; k < DW; k++) if (b_wmask[k]) view[b_addr + k] = b_wdata[32 * k +: 32];
          n_bwr++;
        end else begin
          eb_t e;
          e.tag = b_tag; e.a = b_addr;
          for (int k = 0; k < DW; k++) e.d[32 * k +: 32] = view[b_addr + k];
          eb.push_back(e);
          n_brd++;
        end
      end
      if (a_req && a_rdy) begin
        if (a_we) begin
          for (int k = 0; k < 4; k++) if (a_be[k]) view[a_addr][8 * k +: 8] = a_wdata[8 * k +: 8];
          n_awr++;
        end else begin
          ea_t e;
          e.a = a_addr;
          for (int k = 0; k < AB / 4; k++)
            e.d[32 * k +: 32] = view[(a_addr & ~32'(AB / 4 - 1)) + 32'(k)];
          ea.push_back(e);
          n_ard++;
        end
      end
      if (sw_req && sw_rdy) begin
        for (int k = 0; k < 4; k++) if (sw_be[k]) view[sw_addr][8 * k +: 8] = sw_wdata[8 * k +: 8];
        n_sw++;
      end
      if (w_req && w_rdy) begin
        // logical beat l of the wide chunk on lane (channel) l ^ h, h the chunk's fold; a one-beat
        // read's beat (its chunk's) on its own channel only
        for (int l = 0; l < NCH; l++) begin
          ew_t e;
          logic [31:0] b0;
          b0 = w_one ? w_addr - 32'(CB / 4 * l) : w_addr;
          if (w_one && 32'(l) != (w_addr / (CB / 4)) % NCH) continue;
          e.a = w_addr;
          e.s = w_slot;
          for (int k = 0; k < CB / 4; k++) e.d[32 * k +: 32] = view[b0 + CB / 4 * l + k];
          ew[CL'(l) ^ fold(w_addr)].push_back(e);
        end
        n_wrd++;
        if (w_one) n_one++;
      end
    end
  end

  // the model's memory against the view (once every write is in: wr_idle)
  task automatic check_memory(input int ep);
    int bad;
    bad = 0;
    for (int i = 0; i < WORDS; i++) begin
      logic [31:0] m;
      m = u_mem.mem_lo[i];                // (the model's lower half: WORDS words)
      if (m != view[i]) begin
        if (bad < 8) $display("tb_ndram: epoch %0d word %h: memory %h, expected %h", ep, i, m, view[i]);
        bad++;
      end
    end
    if (bad != 0) $fatal(1, "tb_ndram: epoch %0d: %0d words differ in memory", ep, bad);
  endtask

  task automatic quiesce(input int ep);
    longint t0;
    issuing = 1'b0;
    t0 = cyc;
    do begin
      @(posedge clk);
      if (cyc - t0 > 200000) $fatal(1, "tb_ndram: epoch %0d does not quiesce", ep);
    end while (!(rd_idle && wr_idle && !b_req && !a_req && !sw_req && !w_req &&
                 eb.size() == 0 && w_left() == 0 && ea.size() == 0));
    repeat (2) @(posedge clk);
    if (!(rd_idle && wr_idle)) $fatal(1, "tb_ndram: epoch %0d: idle went away", ep);
    check_memory(ep);
  endtask

  initial begin
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("epochs=%d", epochs));
    void'($value$plusargs("ops=%d", ops));
    void'($value$plusargs("busy=%d", busy));
    void'($value$plusargs("wstream=%d", wstream));
    void'($value$plusargs("bstream=%d", bstream));
    void'($value$plusargs("wcred=%d", wcred));
    void'($value$plusargs("astream=%d", astream));
    void'($value$plusargs("bseq=%d", bseq));
    for (int c = 0; c < NCH; c++) wl_n[c] = 0;
    void'($urandom(seed));
    for (int i = 0; i < WORDS; i++) view[i] = '0;
    repeat (4) @(posedge clk);
    @(negedge clk) rst = 1'b0;
    if (wstream != 0 || bstream != 0 || astream != 0) begin
      longint t0, tb, tw, wb, ta;
      // the memory's content first (B and SW writes), then the stream
      for (int r = 0; r < NR; r++) owner[r] = r % 2 ? O_B : O_SW;
      left_b = ops; left_a = 0; left_sw = ops; left_w = 0;
      @(negedge clk) issuing = 1'b1;
      while (left_b > 0 || left_sw > 0) @(posedge clk);
      quiesce(0);
      for (int r = 0; r < NR; r++) owner[r] = O_RO;
      @(negedge clk);
      left_b = bstream; left_a = astream; left_sw = 0; left_w = wstream;
      a_next = 0; b_next = 0; in_stream = 1'b1;
      t0 = cyc; tb = 0; tw = 0; wb = 0; ta = 0;
      issuing = 1'b1;
      while (left_b > 0 || left_w > 0 || left_a > 0 || eb.size() != 0 || w_left() != 0 ||
             ea.size() != 0) begin
        @(posedge clk);
        if (tb == 0 && left_b == 0 && eb.size() == 0) begin tb = cyc - t0; wb = n_wrd; end
        if (tw == 0 && left_w == 0 && w_left() == 0) tw = cyc - t0;
        if (ta == 0 && left_a == 0 && ea.size() == 0) ta = cyc - t0;
      end
      if (astream != 0)
        $display("TBN ASTREAM reads=%0d cycles=%0d per_cycle=%0.3f", astream, ta,
                 real'(astream) / real'(ta));
      if (bstream != 0)
        $display("TBN BSTREAM reads=%0d cycles=%0d per_cycle=%0.3f w_reads=%0d", bstream, tb,
                 real'(bstream) / real'(tb), wb);
      if (wstream != 0)
        $display("TBN WSTREAM reads=%0d cycles=%0d per_cycle=%0.3f skew=%0d", wstream, tw,
                 real'(wstream) / real'(tw), w_skew);
      quiesce(0);
    end else
      for (int ep = 0; ep < epochs; ep++) begin
        longint t0;
        for (int r = 0; r < NR; r++) owner[r] = rnd(4);
        left_b = ops; left_a = ops; left_sw = ops; left_w = HAS_W ? ops : 0;
        if (ep == 0) begin                // fill first: B and SW (A and W have nothing to read)
          for (int r = 0; r < NR; r++) owner[r] = r % 2 ? O_B : O_SW;
          left_a = 0; left_w = 0;
        end
        @(negedge clk) issuing = 1'b1;
        t0 = cyc;
        while (left_b > 0 || left_a > 0 || left_sw > 0 || left_w > 0) begin
          @(posedge clk);
          if (cyc - t0 > 100 * ops + 200000)
            $fatal(1, "tb_ndram: epoch %0d stuck issuing (B %0d, A %0d, SW %0d, W %0d left)", ep,
                   left_b, left_a, left_sw, left_w);
        end
        quiesce(ep);
      end
    begin
      longint srd, swr, ps;                 // the adapter's SW fill reads, SW writes, masked ones
      srd = 0; swr = 0; ps = 0;
      for (int c = 0; c < NCH; c++) begin
        srd += dut.st_srd[c]; swr += dut.st_swr[c]; ps += dut.st_ps[c];
      end
      $display("TBN DONE cycles=%0d b_rd=%0d b_wr=%0d a_rd=%0d a_wr=%0d sw=%0d w_rd=%0d sw_fill=%0d sw_beats=%0d sw_masked=%0d",
               cyc, n_brd, n_bwr, n_ard, n_awr, n_sw, n_wrd, srd, swr, ps);
      if (WPOST)                            // the cycles B and SW waited on room, on a B beat
        $display("TBN WPOST room_waits=%0d b_waits=%0d", dut.st_cr, dut.st_bh);
      if (XPOST) begin                      // the beats posted (lanes), the one-beat reads
        longint n;
        n = 0;
        for (int c = 0; c < NCH; c++) n += wl_n[c];
        $display("TBN XPOST beats=%0d one_beat_reads=%0d", n, n_one);
      end
    end
    $finish;
  end
endmodule
