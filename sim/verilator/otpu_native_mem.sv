// Simulation model of the board memory behind a slice's NCH channels' native ports (the n_*
// interface of otpu_native_dram, as otpu_mem_ch presents it: one command per CB-byte beat; the
// card's two channels of 64-byte beats, the WG targets' eight of 128), in front of one logical DRAM image
// with otpu_native_dram's channel interleave (with its CHASH: each wide chunk's beats permuted by
// the fold of its index).
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
// per cycle (100 = no limit), or +nat_bpc=N to N / 1000 bytes per core cycle (an HBM channel's
// sustained rate: opentpu.target.mem_model). A refresh-like duty loss: +nat_refi=P +nat_rfc=L
// (cycles; P 0: none) holds each channel's commands for L cycles of every P (channel c's window
// c P / NCH cycles after channel 0's).
//
// DDR3 timing (+axi_dram=1, replaces +axi_bw): a bank model (+axi_map, +axi_trcd +axi_trp
// +axi_tras +axi_trc +axi_trtp +axi_trefi +axi_trfc +axi_tturn, +axi_tpc / +axi_tpu, +axi_trmw
// for a write whose mask is not whole; the names are the MIG AXI path's model's, whose bank model
// this is), with no per-transaction cost (a native command has none). A command is performed once the data bus's backlog is below +nat_ahead=N controller
// cycles, so a busy channel fills the command FIFO and withholds n_cready.
// Images load from dram_<SID>.bin and dump to dram_out_<SID>.bin, as otpu_dram. With PHYS = 1
// the files are the channels' own memories instead, as the host sees them: ch<c>.bin (big-endian
// words, as $fread reads) in, ch<c>_out.bin (little-endian) out, WORDS / NCH words each (c = NCH
// SID .. NCH SID + NCH - 1: slice SID's channels on a multi-slice board). The dump
// prints each channel's reads, writes, DDR3 row opens and partial writes (MEM ch<c> rd=<n> ...).
// +nat_trace=FILE writes every command taken, one line each: core cycle, channel, we, beat (the
// trace sim/verilator/tb_ldc_replay.sv replays through LiteDRAM's controller).
// WFD > 0 (otpu_native_dram WPOST: write data posted, as the WG targets' bridges take it into u_wd):
// each channel's write data goes into a FIFO of WFD beats that is never withheld (n_wready: room,
// a posted beat into a full one is an error), and n_wfree is its free beats from a register, a
// beat taken counted at once and one written counted +nat_wfl=N cycles later (default 3: the
// bridge's pop synchronized back), so never more than are free.
// XPOST (otpu_native_dram XPOST, (a) v2: the channels post port W's beats into the MXU's chunk
// FIFO, as the WG targets' bridges): a read command of source W (n_csrc 3) returns its beat on p_we /
// p_slot (its n_cslot) / p_d instead of n_rvalid, in the same cycle and order it would have,
// and p_cntg counts the beats so posted (gray, mod 2^PCW), stepping with p_we.
// RHB / WHB (otpu_native_dram's: half beats on the native port): a read beat returned on n_rvalid
// goes in two halves, the low one first (n_rdata's low CB x 4 bits; the high one in the next cycle,
// a beat's halves never apart; a posted beat on p_* meanwhile as before), and posted write data
// comes in halves (n_wdata / n_wmask's low half, the low one first), joined into its beat; n_wfree
// and the FIFO's room count a beat from its low half.
// +nat_rstat=1: at the dump, a line a channel (MEMR ch<c> ...) with its returned-on-n_rvalid reads
// (every read but XPOST's posted ones): taken, the most in flight (taken, data not yet back) and
// the most taken in an aligned 256-cycle window (the bridges' read data FIFO's sizing), and the
// cycles a performed read waited for +nat_rcap; and its write beats: taken, the most in a window
// and the most in flight (data taken, not yet counted on n_wdone). +nat_rcap=N: at most N such reads a channel
// performed and not yet returned (the bridges' read data credits, held from the issue to the
// data): the oldest command waits while it is one and N are out, and every command behind it.
// +nat_slat_sid=S +nat_slat=N: slice S's channels alone read N cycles later and count their
// writes done N cycles later (a slice whose adapter sits a die away: the stages its requests
// and responses cross, as a round trip in core cycles). None by default.
// +nat_win=N: every N core cycles of a run (counted from the core's reset falling, as the slice's
// P / Q trace windows with +bucket=N), a line a slice (MEMW s<SID> c=<the window's last cycle>)
// with each channel's counts since the run's start, five a channel (ch<c>=a,b,c,d,e), one a cycle:
// a posted W read performed, another read performed, a write performed, no command to perform
// (none taken before this cycle), a command held (its write data, +nat_rcap, the duty loss, the
// DDR3 model's backlog); a cycle in none of them held a command for the rate. A line at the dump
// too (MEMW ... end=1). For a map of each channel's idle cycles.
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
  parameter bit CHASH = 1'b1,            // otpu_native_dram's address map
  parameter int NCH   = 2,               // channels (otpu_native_dram NCH, CB)
  parameter int CB    = 64,
  parameter int WFD   = 0,               // posted write data: the FIFO's beats (0: withheld at random)
  parameter int WFW   = 6,               // n_wfree's bits
  parameter bit XPOST = 1'b0,            // W reads' beats posted (p_*; see the top)
  parameter int XSW   = 9,
  parameter int PCW   = XSW + 2,
  parameter bit RHB   = 1'b0,            // read data in half beats (see the top)
  parameter bit WHB   = 1'b0,            // write data in half beats
  parameter int AW    = 32               // byte address bits (otpu_native_dram AW): the model
                                         //   holds 4 GiB at most, a beat above it is an error
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  run_rst,    // the core's reset: a rise ends a run (pokeb lines)
  input  logic [NCH-1:0]        n_cvalid,
  output logic [NCH-1:0]        n_cready,
  input  logic [NCH-1:0]        n_cwe,
  input  logic [NCH-1:0][AW-1-$clog2(NCH*CB):0] n_caddr,
  input  logic [NCH-1:0][1:0]   n_csrc,     // XPOST: a read's source (3: W, posted)
  input  logic [NCH-1:0][XSW-1:0] n_cslot,
  input  logic [NCH-1:0]        n_wvalid,
  output logic [NCH-1:0]        n_wready,
  input  logic [NCH-1:0][CB*8-1:0] n_wdata,
  input  logic [NCH-1:0][CB-1:0] n_wmask,
  output logic [NCH-1:0]        n_rvalid,
  output logic [NCH-1:0][CB*8-1:0] n_rdata,
  output logic [NCH-1:0][15:0]  n_wdone,
  output logic [NCH-1:0][WFW-1:0] n_wfree,  // WFD > 0: the write-data FIFO's free beats
  output logic [NCH-1:0]        p_we,       // XPOST: a W beat posted (p_slot's entry, lane c)
  output logic [NCH-1:0][XSW-1:0] p_slot,
  output logic [NCH-1:0][CB*8-1:0] p_d,
  output logic [NCH-1:0][PCW-1:0] p_cntg,   // XPOST: the beats posted (gray)
  input  logic                  dump
);
  localparam int CAW = 32 - $clog2(NCH * CB);
  localparam int BW = CB / 4;            // words a beat
  localparam int CL = $clog2(NCH);
  // the image's words: word i is mem_lo[i] below H, else mem_hi[i - H] (two arrays, as Verilator
  // takes at most 2^28 entries in one: 1 GiB each); a beat (BW words) is in one of them
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
  int bpc = 0;                           // bytes per cycle x 1000 (0: from bw)
  int refi = 0, rfc = 0;                 // the duty loss, cycles
  int lat = LAT;
  int cqd = 16, wqd = 16;                // command / write-data FIFO entries
  int wfl = 3;                           // WFD: cycles until a written beat counts free
  int wdl = 4;                           // n_wdone lag, cycles (+ 0..7)
  int ahead = 16;                        // DDR3 model: data-bus backlog, controller cycles
  longint n_rd [NCH], n_wr [NCH], n_miss [NCH], n_rmw [NCH];
  // DDR3 model
  int dram = 0, amap = 0;
  int trcd = 2, trp = 2, tras = 4, trc = 6, trtp = 1, trefi = 780, trfc = 16, tturn = 2;
  int trmw = 12;                         // core cycles
  int tpc = 4, tpu = 4;                  // ticks per core / controller cycle
  longint cyc = 0;
  string tfn;
  integer tfd = 0;
  initial if ($value$plusargs("nat_trace=%s", tfn)) tfd = $fopen(tfn, "w");
  bit rstat;
  initial rstat = $test$plusargs("nat_rstat");
  int rcap = 0;
  initial void'($value$plusargs("nat_rcap=%d", rcap));
  int win = 0;                           // +nat_win: MEMW lines every win cycles of a run
  initial void'($value$plusargs("nat_win=%d", win));
  longint t0 = 0;                        // the cycle the core's reset last fell
  longint a_pw [NCH], a_pr [NCH], a_pwr [NCH], a_em [NCH], a_bl [NCH];
  longint b_pw [NCH], b_pr [NCH], b_pwr [NCH], b_em [NCH], b_bl [NCH];   // their values at t0

  // the logical beat of channel c's beat m: m's fold into CL bits permutes its wide chunk's beats
  function automatic int lbeat(input logic [31:0] m, input int c);
    logic [CL-1:0] h;
    h = '0;
    if (CHASH) for (int i = 0; i < 32; i++) h[i % CL] ^= m[i];
    return int'(m * NCH + (CL'(c) ^ h));
  endfunction
  function automatic bit rnd_stall();
    return ($urandom % 100) < stall;
  endfunction

  // the controller's per-channel state: the data bus's next free tick, the last direction, the
  // next refresh, each bank's open row (-1: closed), activate time and last column access
  longint bus [NCH], nref [NCH];
  bit     wdir [NCH];
  int     orow [NCH][8];
  longint tact [NCH][8], tcol [NCH][8];
  // the core cycle a beat's column command goes out (and the controller's state after it), with
  // the channel offset m * CB
  function automatic longint dram_slot(input int c, input logic [CAW-1:0] m, input bit wr,
                                       input longint arrive_cyc);
    logic [31:0] off;
    int bk, row;
    longint t, arrive;
    arrive = arrive_cyc * tpc;
    off = 32'(m) * CB;
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

  typedef struct { logic we; logic [CAW-1:0] m; longint t; logic x; logic [XSW-1:0] s; } cq_t;
  typedef struct { longint t; logic [CB*8-1:0] d; logic x; logic [XSW-1:0] s; } rq_t;
  cq_t          cq [NCH][$];             // commands taken, not yet performed
  logic [CB*8-1:0] wq_d [NCH][$];        // write data taken, not yet performed
  logic [CB-1:0] wq_m [NCH][$];
  rq_t          rq [NCH][$];             // read data due
  longint       wd [NCH][$];             // the times writes count in n_wdone

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
  logic rr_w = 1'b1;
  always @(posedge clk) begin
    if (rr_w && !run_rst) begin
      t0 = cyc;
      for (int c = 0; c < NCH; c++) begin
        b_pw[c] = a_pw[c]; b_pr[c] = a_pr[c]; b_pwr[c] = a_pwr[c]; b_em[c] = a_em[c];
        b_bl[c] = a_bl[c];
      end
    end
    rr_w = run_rst;
    if (win != 0 && !run_rst && !rst && (cyc - t0) % win == longint'(win - 1)) memw(0);
    if (win != 0 && dump) memw(1);
  end
  function automatic void memw(input bit e);
    string l;
    l = $sformatf("MEMW s%0d c=%0d", SID, cyc - t0);
    for (int c = 0; c < NCH; c++)
      l = {l, $sformatf(" ch%0d=%0d,%0d,%0d,%0d,%0d", c, a_pw[c] - b_pw[c], a_pr[c] - b_pr[c],
                        a_pwr[c] - b_pwr[c], a_em[c] - b_em[c], a_bl[c] - b_bl[c])};
    if (e) l = {l, " end=1"};
    $display("%s", l);
  endfunction

  for (genvar c = 0; c < NCH; c++) begin : g_ch
    logic crr, wrr, rv, pv;
    logic [CB*8-1:0] rd, pdr;
    bit rh_p;                            // RHB: a read beat's high half goes next (rh_d)
    logic [CB*4-1:0] rh_d;
    bit wl_v;                            // WHB: a write beat's low half is in (wl_d, wl_m)
    logic [CB*4-1:0] wl_d;
    logic [CB/2-1:0] wl_m;
    logic [XSW-1:0] ps;
    logic [PCW-1:0] pn;                  // the beats posted
    logic [15:0] wdone;
    int cr = 0;                          // bandwidth credit (CB x 1000 = one command)
    int nf = 0, nf_max = 0, nw = 0, nw_max = 0;  // +nat_rstat: n_rvalid reads in flight, a window's
    longint n_nr = 0, n_cw = 0, n_wb = 0;
    int ww = 0, ww_max = 0, wfl = 0, wfl_max = 0;   // +nat_rstat: write beats a window, in flight
    int nfp = 0;                         // +nat_rcap: n_rvalid reads performed, data not yet back
    longint rlast = 0, wlast = 0;        // the last read's data time, write's count time
    int cq_n0 = 0;                       // +nat_win: a command was there to perform
    bit cq_hold, did, did_w, did_x;
    always_ff @(posedge clk) begin
      crr <= !rnd_stall();
      wrr <= !rnd_stall();
    end
    assign n_cready[c] = crr && cq[c].size() < cqd;
    assign n_wready[c] = (WHB && wl_v) || (WFD > 0 ? wq_d[c].size() + int'(wl_v) < WFD
                                                    : wrr && wq_d[c].size() < wqd);
    int wocc = 0;                        // WFD: beats taken and not yet counted free
    longint wfr [$];                     //   the times written beats count free
    logic [WFW-1:0] wfree;
    assign n_wfree[c] = wfree;
    assign n_rvalid[c] = rv;
    assign n_rdata[c] = rd;
    assign p_we[c] = pv;
    assign p_slot[c] = ps;
    assign p_d[c] = pdr;
    assign p_cntg[c] = pn ^ (pn >> 1);
    assign n_wdone[c] = wdone;
    always_ff @(posedge clk) begin
      if (rst) begin
        cq[c].delete(); wq_d[c].delete(); wq_m[c].delete(); rq[c].delete(); wd[c].delete();
        rv <= 1'b0; pv <= 1'b0; pn <= '0; wdone <= '0; cr = 0; rlast = 0; wlast = 0;
        rh_p = 1'b0; wl_v = 1'b0;
        nf = 0; nfp = 0; wfl = 0;
        a_pw[c] = 0; a_pr[c] = 0; a_pwr[c] = 0; a_em[c] = 0; a_bl[c] = 0;
        wocc = 0; wfr.delete(); wfree <= '0;
        bus[c] = 0; nref[c] = trefi * tpu; wdir[c] = 1'b0;
        for (int k = 0; k < 8; k++) begin orow[c][k] = -1; tact[c][k] = -1000; tcol[c][k] = -1000; end
      end else begin
        if (n_cvalid[c] && n_cready[c]) begin
          if (AW > 32 && (n_caddr[c] >> CAW) != 0 || lbeat(32'(n_caddr[c]), c) * BW + BW > WORDS)
            $fatal(1, "native command beyond memory");
          cq[c].push_back('{n_cwe[c], n_caddr[c], cyc, XPOST && !n_cwe[c] && n_csrc[c] == 2'd3,
                            n_cslot[c]});
          if (!n_cwe[c] && !(XPOST && n_csrc[c] == 2'd3)) begin nf++; nw++; n_nr++; end
          if (tfd != 0) $fdisplay(tfd, "%0d %0d %0d %0d", cyc, c, n_cwe[c], n_caddr[c]);
        end
        while (wfr.size() != 0 && wfr[0] <= cyc) begin wocc--; void'(wfr.pop_front()); end
        if (WFD > 0 && n_wvalid[c] && !n_wready[c])
          $fatal(1, "otpu_native_mem: channel %0d: a beat posted into a full write-data FIFO", c);
        if (n_wvalid[c] && n_wready[c] && WHB && !wl_v) begin   // (a low half: its beat's room)
          wl_d = n_wdata[c][CB*4-1:0];
          wl_m = n_wmask[c][CB/2-1:0];
          wl_v = 1'b1;
          wocc++;
        end else if (n_wvalid[c] && n_wready[c]) begin
          wq_d[c].push_back(WHB ? {n_wdata[c][CB*4-1:0], wl_d} : n_wdata[c]);
          wq_m[c].push_back(WHB ? {n_wmask[c][CB/2-1:0], wl_m} : n_wmask[c]);
          if (!WHB) wocc++;
          wl_v = 1'b0;
          ww++; wfl++; n_wb++;
        end
        wfree <= WFD > 0 ? WFW'(WFD - wocc) : '0;
        cr = (dram != 0) ? 2000 * CB : (cr + bpc > 2000 * CB) ? 2000 * CB : cr + bpc;
        // perform the oldest command (taken in an earlier cycle): a write once its data is in;
        // none in the channel's refresh window
        if (rcap != 0 && cq[c].size() != 0 && cq[c][0].t < cyc && !cq[c][0].we && !cq[c][0].x &&
            nfp >= rcap)
          n_cw++;
        // (+nat_win) a command to perform this cycle, one held by something but the rate
        cq_n0 = (cq[c].size() != 0 && cq[c][0].t < cyc) ? 1 : 0;
        cq_hold = cq_n0 != 0 && ((cq[c][0].we && wq_d[c].size() == 0) ||
                                 (rcap != 0 && !cq[c][0].we && !cq[c][0].x && nfp >= rcap) ||
                                 (refi != 0 && (cyc + longint'(c * refi / NCH)) % refi < rfc) ||
                                 (dram != 0 && bus[c] > (cyc + longint'(ahead)) * tpc));
        did = 1'b0; did_w = 1'b0; did_x = 1'b0;
        if (cq[c].size() != 0 && cq[c][0].t < cyc && cr >= 1000 * CB &&
            (!cq[c][0].we || wq_d[c].size() != 0) &&
            (rcap == 0 || cq[c][0].we || cq[c][0].x || nfp < rcap) &&
            (refi == 0 || (cyc + longint'(c * refi / NCH)) % refi >= rfc) &&
            (dram == 0 || bus[c] <= (cyc + longint'(ahead)) * tpc)) begin
          int b;
          longint t;
          did = 1'b1; did_w = cq[c][0].we; did_x = cq[c][0].x;
          cr = cr - 1000 * CB;
          b = lbeat(32'(cq[c][0].m), c) * BW;
          if (cq[c][0].we) begin
            logic [CB-1:0] s;
            s = wq_m[c][0];
            for (int k = 0; k < CB; k++)
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
            wfr.push_back(cyc + longint'(wfl));
            n_wr[c]++;
          end else begin
            logic [CB*8-1:0] d;
            for (int k = 0; k < BW; k++) d[32 * k +: 32] = rdw(b + k);
            t = (dram != 0 ? dram_slot(c, cq[c][0].m, 1'b0, cyc) : cyc) + lat + ($urandom % 8);
            if (t < rlast) t = rlast;
            rlast = t;
            rq[c].push_back('{t, d, cq[c][0].x, cq[c][0].s});
            if (!cq[c][0].x) nfp++;
            n_rd[c]++;
          end
          void'(cq[c].pop_front());
        end
        // +nat_win: this cycle's use (performed, by kind; nothing queued; held)
        if (win != 0) begin
          if (cq_n0 == 0) a_em[c]++;
          else if (!did) begin
            if (cq_hold) a_bl[c]++;
          end else if (did_w) a_pwr[c]++;
          else if (did_x) a_pw[c]++;
          else a_pr[c]++;
        end
        // next cycle's read data (no backpressure) and write count
        // (RHB: a beat's high half the cycle after its low half, a posted beat beside it)
        if (RHB && rh_p) begin
          rv <= 1'b1;
          rd <= (CB * 8)'(rh_d);
          rh_p = 1'b0;
          if (rq[c].size() != 0 && rq[c][0].t <= cyc && rq[c][0].x && !rnd_stall()) begin
            pv <= 1'b1;
            pn <= pn + 1'b1;
            pdr <= rq[c][0].d;
            ps <= rq[c][0].s;
            void'(rq[c].pop_front());
          end else pv <= 1'b0;
        end else if (rq[c].size() != 0 && rq[c][0].t <= cyc && !rnd_stall()) begin
          rv <= !rq[c][0].x;
          pv <= rq[c][0].x;
          if (rq[c][0].x) pn <= pn + 1'b1;
          else begin nf--; nfp--; end
          if (RHB && !rq[c][0].x) begin
            rd <= (CB * 8)'(rq[c][0].d[CB*4-1:0]);
            rh_d = rq[c][0].d[CB*8-1:CB*4];
            rh_p = 1'b1;
          end else rd <= rq[c][0].d;
          pdr <= rq[c][0].d;
          ps <= rq[c][0].s;
          void'(rq[c].pop_front());
        end else begin
          rv <= 1'b0;
          pv <= 1'b0;
        end
        if (nf > nf_max) nf_max = nf;
        if (wfl > wfl_max) wfl_max = wfl;
        if (cyc % 256 == 255) begin
          if (nw > nw_max) nw_max = nw;
          if (ww > ww_max) ww_max = ww;
          nw = 0; ww = 0;
        end
        if (!rnd_stall()) begin
          logic [15:0] n;
          n = wdone;
          while (wd[c].size() != 0 && wd[c][0] <= cyc) begin
            n = n + 1'b1;
            void'(wd[c].pop_front());
          end
          wfl = wfl - int'(16'(n - wdone));
          wdone <= n;
        end
      end
    end
    always @(posedge clk)
      if (dump && rstat)
        $display("MEMR ch%0d nr=%0d inflight_max=%0d per256_max=%0d capwait=%0d wr=%0d wr256_max=%0d wr_inflight_max=%0d",
                 NCH * SID + c, n_nr, nf_max, nw_max, n_cw, n_wb, ww_max, wfl_max);
  end

  string dir;
  integer fd, nread;
  logic [31:0] chm [WORDS / NCH];
  initial begin
    void'($value$plusargs("axi_stall=%d", stall));
    void'($value$plusargs("axi_bw=%d", bw));
    bpc = bw * CB * 10;
    void'($value$plusargs("nat_bpc=%d", bpc));
    void'($value$plusargs("nat_refi=%d", refi));
    void'($value$plusargs("nat_rfc=%d", rfc));
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
    void'($value$plusargs("nat_wfl=%d", wfl));
    void'($value$plusargs("nat_wdl=%d", wdl));
    void'($value$plusargs("nat_ahead=%d", ahead));
    begin
      int ss = -1, sl = 0;
      void'($value$plusargs("nat_slat_sid=%d", ss));
      void'($value$plusargs("nat_slat=%d", sl));
      if (ss == SID) begin lat += sl; wdl += sl; end
    end
    for (int c = 0; c < NCH; c++) begin n_rd[c] = 0; n_wr[c] = 0; n_miss[c] = 0; n_rmw[c] = 0; end
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
        // channel c word j (beat j / BW) is logical beat lbeat(j / BW, c)
        for (int c = 0; c < NCH; c++) begin
          for (int i = 0; i < WORDS / NCH; i++) chm[i] = '0;
          fd = $fopen($sformatf("%s/ch%0d.bin", dir, NCH * SID + c), "rb");
          if (fd != 0) begin
            nread = $fread(chm, fd);
            $fclose(fd);
          end
          for (int j = 0; j < WORDS / NCH; j++) wrw(lbeat(j / BW, c) * BW + j % BW, chm[j]);
        end
      end
    end
  end
  always @(posedge clk) if (dump) begin
    for (int c = 0; c < NCH; c++) begin
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
      for (int c = 0; c < NCH; c++) begin
        fd = $fopen($sformatf("%s/ch%0d_out.bin", dir, NCH * SID + c), "wb");
        for (int j = 0; j < WORDS / NCH; j++) $fwrite(fd, "%u", rdw(lbeat(j / BW, c) * BW + j % BW));
        $fclose(fd);
      end
    end
  end
endmodule
