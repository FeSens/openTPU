// Simulation model of the board memory behind the two channels' native ports (the n_* interface
// of otpu_native_dram, as otpu_mem_ch presents it: one command per 64-byte beat), in front of one
// logical DRAM image with otpu_native_dram's channel interleave (with its CHASH: the chunk halves
// swapped by the chunk index's parity).
//
// Per channel, the model takes commands (n_cready) and write data (n_wready) independently, each
// randomly withheld (seed +axi_seed=N, probability +axi_stall=percent) and each into its own FIFO
// (+nat_cq=N / +nat_wq=N entries), so write data is taken before or after its command. It
// performs the commands in command order, at most one a cycle: a write once its data is in (the
// byte mask merges into memory: 1 = write the byte), a read by sampling the memory then. So a
// read after a write to its beat sees the write, and a read before it does not, as the channel
// module promises. Read data returns in order with no backpressure, LAT (+axi_lat) cycles after
// the read is performed plus 0..7 cycles of jitter and random bubbles. n_wdone counts the writes
// performed, +nat_wdl=N cycles later plus 0..7 cycles of jitter and random stalls (so it may step
// by more than 1 a cycle). Bandwidth: +axi_bw=P limits each channel to P percent of one command
// per cycle (100 = no limit).
//
// DDR3 timing (+axi_dram=1, replaces +axi_bw): a bank model (+axi_map, +axi_trcd +axi_trp
// +axi_tras +axi_trc +axi_trtp +axi_trefi +axi_trfc +axi_tturn, +axi_tpc / +axi_tpu, +axi_trmw
// for a write whose mask is not whole; the names are the MIG AXI path's model's, whose bank model
// this is), with no per-transaction cost (a native command has none). A command is performed once the data bus's backlog is below +nat_ahead=N controller
// cycles, so a busy channel fills the command FIFO and withholds n_cready.
// Images load from dram_<SID>.bin and dump to dram_out_<SID>.bin, as otpu_dram. With PHYS = 1
// the files are the channels' own memories instead, as the host sees them: ch<c>.bin (big-endian
// words, as $fread reads) in, ch<c>_out.bin (little-endian) out, WORDS / 2 words each. The dump
// prints each channel's reads, writes, DDR3 row opens and partial writes (MEM ch<c> rd=<n> ...).
// +nat_trace=FILE writes every command taken, one line each: core cycle, channel, we, beat (the
// trace sim/verilator/tb_ldc_replay.sv replays through LiteDRAM's controller).
// The host's writes come through its own master (XDMA on the card), not the n_* port, so the
// adapter does not see them: <dir>/poke_<SID>.txt, lines "cycle word value" (hex; the logical
// word address), written at that core cycle (a run's: WAITW's tests); <dir>/pokeb_<SID>.txt,
// lines "run word value", written when run `run` - 1 ends (run_rst, the core's reset, rises: tb_top
// +runs).
module otpu_native_mem #(
  parameter int WORDS = 1 << 18,
  parameter int PHYS  = 0,
  parameter int LAT   = 20,              // minimum read latency (performed -> data)
  parameter int SID   = 0,
  parameter bit CHASH = 1'b1             // otpu_native_dram's address map
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  run_rst,    // the core's reset: a rise ends a run (pokeb lines)
  input  logic [1:0]            n_cvalid,
  output logic [1:0]            n_cready,
  input  logic [1:0]            n_cwe,
  input  logic [1:0][24:0]      n_caddr,
  input  logic [1:0]            n_wvalid,
  output logic [1:0]            n_wready,
  input  logic [1:0][511:0]     n_wdata,
  input  logic [1:0][63:0]      n_wmask,
  output logic [1:0]            n_rvalid,
  output logic [1:0][511:0]     n_rdata,
  output logic [1:0][15:0]      n_wdone,
  input  logic                  dump
);
  // the image's words: word i is mem_lo[i] below H, else mem_hi[i - H] (two arrays, as Verilator
  // takes at most 2^28 entries in one: 1 GiB each); a beat (16 words) is in one of them
  localparam int H = WORDS / 2;
  logic [31:0] mem_lo [H];
  logic [31:0] mem_hi [WORDS - H];
  function automatic logic [31:0] rdw(input int i);
    return i < H ? mem_lo[i] : mem_hi[i - H];
  endfunction
  task automatic wrw(input int i, input logic [31:0] v);
    if (i < H) mem_lo[i] = v;
    else mem_hi[i - H] = v;
  endtask
  int stall = 0;
  int bw = 100;
  int lat = LAT;
  int cqd = 16, wqd = 16;                // command / write-data FIFO entries
  int wdl = 4;                           // n_wdone lag, cycles (+ 0..7)
  int ahead = 16;                        // DDR3 model: data-bus backlog, controller cycles
  longint n_rd [2], n_wr [2], n_miss [2], n_rmw [2];
  // DDR3 model
  int dram = 0, amap = 0;
  int trcd = 2, trp = 2, tras = 4, trc = 6, trtp = 1, trefi = 780, trfc = 16, tturn = 2;
  int trmw = 12;                         // core cycles
  int tpc = 4, tpu = 4;                  // ticks per core / controller cycle
  longint cyc = 0;
  string tfn;
  integer tfd = 0;
  initial if ($value$plusargs("nat_trace=%s", tfn)) tfd = $fopen(tfn, "w");

  // the logical beat of channel c's beat m
  function automatic int lbeat(input logic [31:0] m, input int c);
    return int'(m * 2 + (c[0] ^ (CHASH && ^m)));
  endfunction
  function automatic bit rnd_stall();
    return ($urandom % 100) < stall;
  endfunction

  // the controller's per-channel state: the data bus's next free tick, the last direction, the
  // next refresh, each bank's open row (-1: closed), activate time and last column access
  longint bus [2], nref [2];
  bit     wdir [2];
  int     orow [2][8];
  longint tact [2][8], tcol [2][8];
  // the core cycle a beat's column command goes out (and the controller's state after it), with
  // the channel offset m * 64
  function automatic longint dram_slot(input int c, input logic [24:0] m, input bit wr,
                                       input longint arrive_cyc);
    logic [30:0] off;
    int bk, row;
    longint t, arrive;
    arrive = arrive_cyc * tpc;
    off = {m, 6'd0};
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

  typedef struct { logic we; logic [24:0] m; longint t; } cq_t;
  typedef struct { longint t; logic [511:0] d; } rq_t;
  cq_t          cq [2][$];               // commands taken, not yet performed
  logic [511:0] wq_d [2][$];             // write data taken, not yet performed
  logic [63:0]  wq_m [2][$];
  rq_t          rq [2][$];               // read data due
  longint       wd [2][$];               // the times writes count in n_wdone

  always_ff @(posedge clk) cyc <= cyc + 1;

  localparam int NPOKE = 4096;
  longint      pk_c [NPOKE];
  logic [31:0] pk_a [NPOKE], pk_v [NPOKE], pb_r [NPOKE], pb_a [NPOKE], pb_v [NPOKE];
  int          pk_n = 0, pk_i = 0, pb_n = 0;
  int          epoch = 0;                // runs ended
  logic        rr_q = 1'b1;
  always @(posedge clk) begin
    while (pk_i < pk_n && cyc >= pk_c[pk_i]) begin
      wrw(int'(pk_a[pk_i]), pk_v[pk_i]);
      pk_i++;
    end
    if (run_rst && !rr_q) begin          // a run ended: the host's writes before the next
      epoch++;
      for (int k = 0; k < pb_n; k++) if (int'(pb_r[k]) == epoch) wrw(int'(pb_a[k]), pb_v[k]);
    end
    rr_q = run_rst;
  end

  for (genvar c = 0; c < 2; c++) begin : g_ch
    logic crr, wrr, rv;
    logic [511:0] rd;
    logic [15:0] wdone;
    int cr = 0;                          // bandwidth credit (100 = one command)
    longint rlast = 0, wlast = 0;        // the last read's data time, write's count time
    always_ff @(posedge clk) begin
      crr <= !rnd_stall();
      wrr <= !rnd_stall();
    end
    assign n_cready[c] = crr && cq[c].size() < cqd;
    assign n_wready[c] = wrr && wq_d[c].size() < wqd;
    assign n_rvalid[c] = rv;
    assign n_rdata[c] = rd;
    assign n_wdone[c] = wdone;
    always_ff @(posedge clk) begin
      if (rst) begin
        cq[c].delete(); wq_d[c].delete(); wq_m[c].delete(); rq[c].delete(); wd[c].delete();
        rv <= 1'b0; wdone <= '0; cr = 0; rlast = 0; wlast = 0;
        bus[c] = 0; nref[c] = trefi * tpu; wdir[c] = 1'b0;
        for (int k = 0; k < 8; k++) begin orow[c][k] = -1; tact[c][k] = -1000; tcol[c][k] = -1000; end
      end else begin
        if (n_cvalid[c] && n_cready[c]) begin
          if (lbeat(32'(n_caddr[c]), c) * 16 + 16 > WORDS)
            $fatal(1, "native command beyond memory");
          cq[c].push_back('{n_cwe[c], n_caddr[c], cyc});
          if (tfd != 0) $fdisplay(tfd, "%0d %0d %0d %0d", cyc, c, n_cwe[c], n_caddr[c]);
        end
        if (n_wvalid[c] && n_wready[c]) begin
          wq_d[c].push_back(n_wdata[c]);
          wq_m[c].push_back(n_wmask[c]);
        end
        cr = (dram != 0) ? 200 : (cr + bw > 200) ? 200 : cr + bw;
        // perform the oldest command (taken in an earlier cycle): a write once its data is in
        if (cq[c].size() != 0 && cq[c][0].t < cyc && cr >= 100 &&
            (!cq[c][0].we || wq_d[c].size() != 0) &&
            (dram == 0 || bus[c] <= (cyc + longint'(ahead)) * tpc)) begin
          int b;
          longint t;
          cr = cr - 100;
          b = lbeat(32'(cq[c][0].m), c) * 16;
          if (cq[c][0].we) begin
            logic [63:0] s;
            s = wq_m[c][0];
            for (int k = 0; k < 64; k++)
              if (s[k] && b < H) mem_lo[b + k / 4][8 * (k % 4) +: 8] <= wq_d[c][0][8 * k +: 8];
              else if (s[k]) mem_hi[b - H + k / 4][8 * (k % 4) +: 8] <= wq_d[c][0][8 * k +: 8];
            t = cyc;
            if (dram != 0) begin
              if (s != '1) t = dram_slot(c, cq[c][0].m, 1'b0, cyc) + trmw;
              t = dram_slot(c, cq[c][0].m, 1'b1, t);
            end
            if (s != '1) n_rmw[c]++;
            t = t + wdl + ($urandom % 8);
            if (t < wlast) t = wlast;
            wlast = t;
            wd[c].push_back(t);
            void'(wq_d[c].pop_front()); void'(wq_m[c].pop_front());
            n_wr[c]++;
          end else begin
            logic [511:0] d;
            for (int k = 0; k < 16; k++) d[32 * k +: 32] = rdw(b + k);
            t = (dram != 0 ? dram_slot(c, cq[c][0].m, 1'b0, cyc) : cyc) + lat + ($urandom % 8);
            if (t < rlast) t = rlast;
            rlast = t;
            rq[c].push_back('{t, d});
            n_rd[c]++;
          end
          void'(cq[c].pop_front());
        end
        // next cycle's read data (no backpressure) and write count
        if (rq[c].size() != 0 && rq[c][0].t <= cyc && !rnd_stall()) begin
          rv <= 1'b1;
          rd <= rq[c][0].d;
          void'(rq[c].pop_front());
        end else begin
          rv <= 1'b0;
        end
        if (!rnd_stall()) begin
          logic [15:0] n;
          n = wdone;
          while (wd[c].size() != 0 && wd[c][0] <= cyc) begin
            n = n + 1'b1;
            void'(wd[c].pop_front());
          end
          wdone <= n;
        end
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
    void'($value$plusargs("axi_tpc=%d", tpc));
    void'($value$plusargs("axi_tpu=%d", tpu));
    void'($value$plusargs("nat_cq=%d", cqd));
    void'($value$plusargs("nat_wq=%d", wqd));
    void'($value$plusargs("nat_wdl=%d", wdl));
    void'($value$plusargs("nat_ahead=%d", ahead));
    n_rd = '{0, 0}; n_wr = '{0, 0}; n_miss = '{0, 0}; n_rmw = '{0, 0};
    begin
      int seed;
      if ($value$plusargs("axi_seed=%d", seed)) void'($urandom(seed));
    end
    for (int i = 0; i < H; i++) mem_lo[i] = '0;
    for (int i = 0; i < WORDS - H; i++) mem_hi[i] = '0;
    if ($value$plusargs("dir=%s", dir)) begin
      if (PHYS == 0) begin
        fd = $fopen($sformatf("%s/dram_%0d.bin", dir, SID), "rb");
        if (fd != 0) begin
          nread = $fread(mem_lo, fd);  // the second continues where the first stopped
          nread = $fread(mem_hi, fd);
          $fclose(fd);
        end
        fd = $fopen($sformatf("%s/poke_%0d.txt", dir, SID), "r");
        if (fd != 0) begin
          while (pk_n < NPOKE && $fscanf(fd, "%h %h %h", pk_c[pk_n], pk_a[pk_n], pk_v[pk_n]) == 3)
            pk_n++;
          $fclose(fd);
        end
        fd = $fopen($sformatf("%s/pokeb_%0d.txt", dir, SID), "r");
        if (fd != 0) begin
          while (pb_n < NPOKE && $fscanf(fd, "%h %h %h", pb_r[pb_n], pb_a[pb_n], pb_v[pb_n]) == 3)
            pb_n++;
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
          for (int j = 0; j < WORDS / 2; j++) wrw(lbeat(j / 16, c) * 16 + j % 16, chm[j]);
        end
      end
    end
  end
  always @(posedge clk) if (dump) begin
    for (int c = 0; c < 2; c++) begin
      int nw;
      nw = 0;
      for (int k = 0; k < cq[c].size(); k++) if (cq[c][k].we) nw++;
      if (nw != 0 || wq_d[c].size() != 0)
        $display("MEM ch%0d: %0d write commands and %0d write beats not performed at the dump",
                 c, nw, wq_d[c].size());
      $display("MEM ch%0d rd=%0d wr=%0d row_miss=%0d rmw=%0d", c, n_rd[c], n_wr[c], n_miss[c],
               n_rmw[c]);
    end
    if (PHYS == 0) begin
      fd = $fopen($sformatf("%s/dram_out_%0d.bin", dir, SID), "wb");
      for (int i = 0; i < WORDS; i++) $fwrite(fd, "%u", rdw(i));
      $fclose(fd);
    end else begin
      for (int c = 0; c < 2; c++) begin
        fd = $fopen($sformatf("%s/ch%0d_out.bin", dir, c), "wb");
        for (int j = 0; j < WORDS / 2; j++) $fwrite(fd, "%u", rdw(lbeat(j / 16, c) * 16 + j % 16));
        $fclose(fd);
      end
    end
  end
endmodule
