// Simulation model of the board memory: two AXI4 slave channels (512-bit; single-beat writes,
// INCR read bursts, which must not cross 4 KB) in front of one logical DRAM image with the 64-byte channel interleave of
// otpu_axi_dram (with its CHASH: the chunk halves swapped by the chunk index's parity). Every ready is randomly withheld and every response randomly delayed (seed
// +axi_seed=N, stall probability +axi_stall=percent), so the slice sees variable latency and
// backpressure; reads and writes are not ordered against each other, as in a real controller:
// a read returns the memory as it was when its AR was accepted, and a write lands when its
// response goes out.
// Bandwidth: +axi_bw=P limits each channel to P percent of one 64-byte beat per cycle (reads and
// writes together; 100 = no limit); +axi_lat=N overrides the minimum latency. +axi_arc=N: a
// cost per read transaction, as the board's interconnect and controller have: a read's data
// starts no sooner than N cycles after the previous read's on its channel (0 = none; single
// 64-byte reads then reach at most 1/N beats per cycle). The final dump prints each channel's
// read transactions and beats (AXI ch<c> ar=<n> beats=<n> ...).
// DDR3 timing (+axi_dram=1, replaces +axi_bw): each channel is one DDR3 rank behind a
// first-come first-served controller, in controller clock cycles (= core cycles: the MIG's
// ui_clk at DDR3-800 4:1 is 100 MHz). A 64-byte beat is one BL8 burst: one cycle of the data
// bus. Every bank keeps its row open; a beat to another row in the same bank precharges (after
// tRTP from the bank's last column access and tRAS from its activate), activates (tRP later, at
// least tRC after the previous activate) and waits tRCD. A bank can be activated as soon as its
// request has arrived and the bank is free (the controller's bank machines work ahead of the
// data bus). Refresh every tREFI blocks the bus for tRFC and closes every row. A read after a
// write or a write after a read loses tTURN; a write with a partial strobe is a read-modify-write
// (the MIG's ECC): its write waits +axi_trmw cycles after the read. Address map +axi_map=0 BANK_ROW_COLUMN (the MIG's default: the bank is the
// top 3 bits of the channel's 2 GB, so a 256 MB region is one bank) or 1 ROW_BANK_COLUMN
// (consecutive 8 KB rows rotate over the 8 banks). +axi_tgap=N: the data bus loses N / 100 controller
// cycles per read transaction (the MIG's AXI front end: on the card, 8-beat bursts deliver
// about 8 / 10 of the peak, see docs/board.md); +axi_wgap=N likewise per write transaction. Times: +axi_trcd +axi_trp +axi_tras
// +axi_trc +axi_trtp +axi_trefi +axi_trfc +axi_tturn (controller cycles); +axi_tpc / +axi_tpu the
// core and controller clock periods in ticks (4 / 4 at DDR3-800, 4 / 3 at DDR3-1066: ui_clk
// 133 MHz, one beat per 0.75 core cycles). What-if: +axi_afree=1 serves port A reads (arid 1)
// without touching the DDR3 timing. +axi_rout=N / +axi_wout=N: at most N read / write
// transactions outstanding per channel (AR / AW accepted, last R / B not yet sent; 0 = no
// limit), as the interconnect's NUM_READ_OUTSTANDING / NUM_WRITE_OUTSTANDING (vivado/bd.tcl).
// Images load from dram_<SID>.bin and dump to dram_out_<SID>.bin, as otpu_dram. With PHYS = 1
// the files are the channels' own memories instead, as the host sees them: ch<c>.bin (big-endian
// words, as $fread reads) in, ch<c>_out.bin (little-endian) out, WORDS / 2 words each.
module otpu_axi_mem #(
  parameter int WORDS = 1 << 18,
  parameter int PHYS  = 0,
  parameter int LAT   = 20,              // minimum read / write-response latency
  parameter int SID   = 0,
  parameter bit CHASH = 1'b1,            // otpu_axi_dram's address map
  parameter logic [31:0] BASE0 = 32'h0000_0000,
  parameter logic [31:0] BASE1 = 32'h8000_0000
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic [1:0]            s_awvalid,
  output logic [1:0]            s_awready,
  input  logic [1:0][31:0]      s_awaddr,
  input  logic [1:0]            s_awid,
  input  logic [1:0]            s_wvalid,
  output logic [1:0]            s_wready,
  input  logic [1:0][511:0]     s_wdata,
  input  logic [1:0][63:0]      s_wstrb,
  output logic [1:0]            s_bvalid,
  input  logic [1:0]            s_bready,
  output logic [1:0]            s_bid,
  output logic [1:0][1:0]       s_bresp,
  input  logic [1:0]            s_arvalid,
  output logic [1:0]            s_arready,
  input  logic [1:0][31:0]      s_araddr,
  input  logic [1:0][7:0]       s_arlen,
  input  logic [1:0]            s_arid,
  output logic [1:0]            s_rvalid,
  input  logic [1:0]            s_rready,
  output logic [1:0]            s_rid,
  output logic [1:0][511:0]     s_rdata,
  output logic [1:0][1:0]       s_rresp,
  output logic [1:0]            s_rlast,
  input  logic                  dump
);
  logic [31:0] mem [WORDS];
  int stall = 0;
  int bw = 100;
  int lat = LAT;
  int arc = 0;
  longint n_ar [2], n_rb [2], n_ara [2], n_miss [2], n_rmw [2];
  // DDR3 model (see the top)
  int dram = 0, amap = 0;
  int trcd = 2, trp = 2, tras = 4, trc = 6, trtp = 1, trefi = 780, trfc = 16, tturn = 2;
  int trmw = 12;                         // core cycles
  int tgap = 0;                          // the data bus's loss per read transaction, 1/100 controller cycle
  int wgap = 0;                          // likewise per write transaction
  int tpc = 4, tpu = 4;                  // ticks per core / controller cycle
  int afree = 0;                         // what-if: port A reads cost the DRAM nothing
  int rout = 0, wout = 0;                // outstanding transaction limits (0: none)
  longint n_rmw_a [2];
  longint cyc = 0;

  // the logical beat of channel c's beat m
  function automatic int lbeat(input logic [31:0] m, input int c);
    return int'(m * 2 + (c[0] ^ (CHASH && ^m)));
  endfunction
  function automatic int beat_word(input logic [31:0] addr, input int c);
    logic [31:0] off;
    off = addr - (c ? BASE1 : BASE0);
    return lbeat(off >> 6, c) * 16;
  endfunction
  function automatic bit rnd_stall();
    return ($urandom % 100) < stall;
  endfunction

  // the controller's per-channel state: the data bus's next free cycle, the last direction,
  // the next refresh, each bank's open row (-1: closed), activate time and last column access
  longint bus [2], nref [2];
  bit     wdir [2];
  int     orow [2][8];
  longint tact [2][8], tcol [2][8];
  // the core cycle a beat's column command goes out (and the controller's state after it). The
  // controller runs in ticks: tpc per core cycle, tpu per controller (ui_clk) cycle, and the DDR3
  // times are in controller cycles (DDR3-800: 4 / 4, ui_clk = core clock; DDR3-1066: 4 / 3)
  function automatic longint dram_slot(input int c, input logic [31:0] addr, input bit wr,
                                       input longint arrive_cyc);
    logic [31:0] off;
    int bk, row;
    longint t, arrive;
    arrive = arrive_cyc * tpc;
    off = addr - (c ? BASE1 : BASE0);
    if (amap == 0) begin bk = int'(off[30:28]); row = int'(off[27:13]); end
    else           begin bk = int'(off[15:13]); row = int'(off[30:16]); end
    t = arrive > bus[c] ? arrive : bus[c];
    while (t >= nref[c]) begin
      if (bus[c] < nref[c]) bus[c] = nref[c];
      bus[c] = bus[c] + trfc * tpu;
      for (int k = 0; k < 8; k++) begin orow[c][k] = -1; tact[c][k] = bus[c] - trc * tpu; end
      nref[c] = nref[c] + trefi * tpu;
      t = arrive > bus[c] ? arrive : bus[c];
    end
    if (wr != wdir[c]) begin t = t + tturn * tpu; wdir[c] = wr; end
    if (orow[c][bk] != row) begin
      longint tp, ta;
      if (orow[c][bk] < 0) tp = arrive - trp * tpu;
      else begin
        tp = tcol[c][bk] + trtp * tpu;
        if (tact[c][bk] + tras * tpu > tp) tp = tact[c][bk] + tras * tpu;
        if (arrive > tp) tp = arrive;
      end
      ta = tp + trp * tpu;
      if (tact[c][bk] + trc * tpu > ta) ta = tact[c][bk] + trc * tpu;
      orow[c][bk] = row;
      tact[c][bk] = ta;
      if (ta + trcd * tpu > t) t = ta + trcd * tpu;
      n_miss[c]++;
    end
    tcol[c][bk] = t;
    bus[c] = t + tpu;
    return (t + tpc - 1) / tpc;
  endfunction

  typedef struct { longint t; logic id; logic [31:0] addr; int len; } rq_t;
  longint rbt [2][$];                    // DDR3 model: each read beat's data time
  typedef struct { longint t; logic id; int b; logic [511:0] d; logic [63:0] s; } bq_t;
  logic [511:0] rdq [2][$];              // each read beat's data, taken when its AR is accepted
  rq_t rq [2][$];
  bq_t bq [2][$];
  logic [31:0] aw_a [2][$];
  logic        aw_i [2][$];
  logic [511:0] w_d [2][$];
  logic [63:0]  w_s [2][$];

  always_ff @(posedge clk) cyc <= cyc + 1;

  for (genvar c = 0; c < 2; c++) begin : g_ch
    logic arr, awr, wr, rv, bv;
    int cr = 0;                          // bandwidth credit (100 = one beat)
    int ri = 0;                          // the head read's next beat
    longint art = 0;                     // the last read's earliest data (transaction cost)
    longint aa = 0;                      // DDR3 model: the last read's arrival at the controller
    always_ff @(posedge clk) begin
      arr <= !rnd_stall();
      awr <= !rnd_stall();
      wr  <= !rnd_stall();
    end
    int nw_out = 0;                      // writes accepted (AW), response not yet taken
    assign s_arready[c] = arr && (rout == 0 || rq[c].size() < rout);
    assign s_awready[c] = awr && (wout == 0 || nw_out < wout);
    assign s_wready[c] = wr;
    // R: the head read, once its time has come (in order per channel)
    always_comb begin
      s_rvalid[c] = 1'b0; s_rid[c] = 1'b0; s_rdata[c] = '0;
      s_rresp[c] = 2'b00; s_rlast[c] = 1'b0;
      if (rv) begin
        s_rvalid[c] = 1'b1;
        s_rid[c] = rq[c][0].id;
        s_rlast[c] = ri == rq[c][0].len - 1;
        s_rdata[c] = rdq[c][0];
      end
      s_bvalid[c] = bv;
      s_bid[c] = bv ? bq[c][0].id : 1'b0;
      s_bresp[c] = 2'b00;
    end
    always_ff @(posedge clk) begin
      if (rst) begin
        rq[c].delete(); bq[c].delete(); aw_a[c].delete(); aw_i[c].delete(); rdq[c].delete();
        w_d[c].delete(); w_s[c].delete();
        rv <= 1'b0; bv <= 1'b0; ri = 0; art = 0; aa = 0; nw_out = 0;
        rbt[c].delete(); bus[c] = 0; nref[c] = trefi * tpu; wdir[c] = 1'b0;
        for (int k = 0; k < 8; k++) begin orow[c][k] = -1; tact[c][k] = -1000; tcol[c][k] = -1000; end
      end else begin
        if (s_arvalid[c] && s_arready[c]) begin
          longint t;
          int n;
          n = int'(s_arlen[c]) + 1;
          if (s_araddr[c][5:0] != 0) $fatal(1, "AXI read not beat aligned");
          if ((s_araddr[c] & 32'hfff) + 32'(64 * n) > 32'h1000) $fatal(1, "AXI read burst crosses 4 KB");
          if (beat_word(s_araddr[c], c) + 32 * (n - 1) + 16 > WORDS) $fatal(1, "AXI read beyond memory");
          t = cyc + lat + ($urandom % 8);
          if (art + arc > t) t = art + arc;
          art = t;
          if (dram != 0) begin
            // the transaction reaches the controller at cyc (+ the per-transaction cost); its
            // beats' data come lat after their column commands
            longint a;
            a = (aa + arc > cyc) ? aa + arc : cyc;
            aa = a;
            if (afree == 0 || !s_arid[c]) begin
              if (bus[c] < a * tpc) bus[c] = a * tpc;
              bus[c] = bus[c] + (tgap * tpu + 50) / 100;
            end
            for (int i = 0; i < n; i++)
              rbt[c].push_back((afree != 0 && s_arid[c] ? a + i
                                : dram_slot(c, s_araddr[c] + 32'(64 * i), 1'b0, a)) + lat);
          end
          rq[c].push_back('{t, s_arid[c], s_araddr[c], n});
          for (int i = 0; i < n; i++) begin
            logic [511:0] d;
            for (int k = 0; k < 16; k++) d[32 * k +: 32] = mem[beat_word(s_araddr[c] + 32'(64 * i), c) + k];
            rdq[c].push_back(d);
          end
          n_ar[c]++;
          n_rb[c] += n;
          if (s_arid[c]) n_ara[c]++;
        end
        if (s_awvalid[c] && s_awready[c]) nw_out = nw_out + 1;
        if (bv && s_bready[c]) nw_out = nw_out - 1;
        if (s_awvalid[c] && s_awready[c]) begin
          aw_a[c].push_back(s_awaddr[c]);
          aw_i[c].push_back(s_awid[c]);
        end
        if (s_wvalid[c] && s_wready[c]) begin
          w_d[c].push_back(s_wdata[c]);
          w_s[c].push_back(s_wstrb[c]);
        end
        cr = (dram != 0) ? 200 : (cr + bw > 200) ? 200 : cr + bw;
        // a write is performed once both its address and data are in (and the channel has time)
        if (aw_a[c].size() != 0 && w_d[c].size() != 0 && cr >= 100) begin
          int b;
          longint tw;
          cr = cr - 100;
          tw = cyc;
          if (dram != 0) begin
            tw = cyc;
            if (w_s[c][0] != '1) begin
              tw = dram_slot(c, aw_a[c][0], 1'b0, cyc) + trmw;
              n_rmw[c]++;
              if (aw_i[c][0]) n_rmw_a[c]++;
            end
            if (bus[c] < tw * tpc) bus[c] = tw * tpc;
            bus[c] = bus[c] + (wgap * tpu + 50) / 100;
            tw = dram_slot(c, aw_a[c][0], 1'b1, tw);
          end
          b = beat_word(aw_a[c][0], c);
          if (b + 16 > WORDS) $fatal(1, "AXI write beyond memory");
          // the write lands in memory when its response goes out: a read accepted before
          // then may or may not see it, as in a real controller (here: never)
          bq[c].push_back('{tw + lat + ($urandom % 8), aw_i[c][0], b, w_d[c][0], w_s[c][0]});
          void'(aw_a[c].pop_front()); void'(aw_i[c].pop_front());
          void'(w_d[c].pop_front()); void'(w_s[c].pop_front());
        end
        if (rv && s_rready[c]) begin
          if (dram != 0) void'(rbt[c].pop_front());
          void'(rdq[c].pop_front());
          if (ri == rq[c][0].len - 1) begin
            void'(rq[c].pop_front());
            ri = 0;
          end else ri++;
        end
        if (bv && s_bready[c]) begin
          for (int k = 0; k < 64; k++)
            if (bq[c][0].s[k]) mem[bq[c][0].b + k / 4][8 * (k % 4) +: 8] <= bq[c][0].d[8 * k +: 8];
          void'(bq[c].pop_front());
        end
        // next cycle's responses (the queues above are already updated)
        if (rq[c].size() != 0 && (dram != 0 ? rbt[c][0] <= cyc : rq[c][0].t <= cyc) &&
            !rnd_stall() && cr >= 100) begin
          rv <= 1'b1;
          if (dram == 0) cr = cr - 100;
        end else begin
          rv <= 1'b0;
        end
        bv <= bq[c].size() != 0 && bq[c][0].t <= cyc && !rnd_stall();
      end
    end
  end

  string dir;
  integer fd, nread;
  logic [31:0] chm [WORDS / 2];
  initial begin
    void'($value$plusargs("axi_stall=%d", stall));
    void'($value$plusargs("axi_bw=%d", bw));
    void'($value$plusargs("axi_lat=%d", lat));
    void'($value$plusargs("axi_arc=%d", arc));
    void'($value$plusargs("axi_dram=%d", dram));
    void'($value$plusargs("axi_map=%d", amap));
    void'($value$plusargs("axi_trcd=%d", trcd));
    void'($value$plusargs("axi_trp=%d", trp));
    void'($value$plusargs("axi_tras=%d", tras));
    void'($value$plusargs("axi_trc=%d", trc));
    void'($value$plusargs("axi_trtp=%d", trtp));
    void'($value$plusargs("axi_trefi=%d", trefi));
    void'($value$plusargs("axi_trfc=%d", trfc));
    void'($value$plusargs("axi_tturn=%d", tturn));
    void'($value$plusargs("axi_trmw=%d", trmw));
    void'($value$plusargs("axi_tgap=%d", tgap));
    void'($value$plusargs("axi_wgap=%d", wgap));
    void'($value$plusargs("axi_tpc=%d", tpc));
    void'($value$plusargs("axi_tpu=%d", tpu));
    void'($value$plusargs("axi_afree=%d", afree));
    void'($value$plusargs("axi_rout=%d", rout));
    void'($value$plusargs("axi_wout=%d", wout));
    n_ar = '{0, 0}; n_rb = '{0, 0}; n_ara = '{0, 0}; n_miss = '{0, 0}; n_rmw = '{0, 0}; n_rmw_a = '{0, 0};
    begin
      int seed;
      if ($value$plusargs("axi_seed=%d", seed)) void'($urandom(seed));
    end
    for (int i = 0; i < WORDS; i++) mem[i] = '0;
    if ($value$plusargs("dir=%s", dir)) begin
      if (PHYS == 0) begin
        fd = $fopen($sformatf("%s/dram_%0d.bin", dir, SID), "rb");
        if (fd != 0) begin
          nread = $fread(mem, fd);
          $fclose(fd);
        end
      end else begin
        // channel c word j (beat j / 16) is logical beat lbeat(j / 16, c)
        for (int c = 0; c < 2; c++) begin
          for (int i = 0; i < WORDS / 2; i++) chm[i] = '0;
          fd = $fopen($sformatf("%s/ch%0d.bin", dir, c), "rb");
          if (fd != 0) begin
            nread = $fread(chm, fd);
            $fclose(fd);
          end
          for (int j = 0; j < WORDS / 2; j++) mem[lbeat(j / 16, c) * 16 + j % 16] = chm[j];
        end
      end
    end
  end
  always @(posedge clk) if (dump) begin
    for (int c = 0; c < 2; c++)
      $display("AXI ch%0d ar=%0d beats=%0d ar_a=%0d row_miss=%0d rmw=%0d rmw_a=%0d", c, n_ar[c],
               n_rb[c], n_ara[c], n_miss[c], n_rmw[c], n_rmw_a[c]);
    if (PHYS == 0) begin
      fd = $fopen($sformatf("%s/dram_out_%0d.bin", dir, SID), "wb");
      for (int i = 0; i < WORDS; i++) $fwrite(fd, "%u", mem[i]);
      $fclose(fd);
    end else begin
      for (int c = 0; c < 2; c++) begin
        fd = $fopen($sformatf("%s/ch%0d_out.bin", dir, c), "wb");
        for (int j = 0; j < WORDS / 2; j++) $fwrite(fd, "%u", mem[lbeat(j / 16, c) * 16 + j % 16]);
        $fclose(fd);
      end
    end
  end
endmodule
