// Slice DRAM ports (as otpu_dram) on the two DDR3 channels' native ports, 512-bit data: one
// command per 64-byte beat (LiteDRAM's native port behind otpu_mem_ch; docs/litedram.md section
// 3). The slice side, the address map and the behaviour are those of otpu_axi_dram, the MIG AXI
// builds' adapter it replaced. What existed only for AXI and the MIG's AXI front end, whose
// transactions cost a lot, is gone: read and write bursts, their gather timers and 4 KB splits,
// AXI IDs, write responses and err. A native command costs nothing per transaction, so a request
// goes out as soon as it may, one command per beat.
//
// Address map: the slice's byte address space is interleaved over the channels in 64-byte beats:
// logical beat b = addr / 64 (chunk m = b / 2) is channel beat m (n_caddr: the channel address's
// bits 30:6, 2 GiB per channel) on channel c = b % 2, or with CHASH on c = (b % 2) ^ parity(m). A
// D = 128-byte chunk (port B; requests are chunk aligned) is therefore one beat on each channel,
// and a streamed operand uses both channels evenly. CHASH swaps the chunks' halves by the parity
// of the chunk index, so the beats at a power-of-two stride of chunks (a transposed V's column:
// one byte per cache row) fall half on each channel instead of all on one. The host applies the
// same map when it fills and reads the memory (opentpu/host/board.py; CAPS bit7 = CHASH).
//
// The native port, per channel (core clock): a command stream (we, beat); one write-data beat
// (data, byte mask with 1 = write the byte) per write command, in write-command order, which may
// be taken before or after its command; read data in read-command order with no backpressure;
// n_wdone, the count of write beats the controller has taken. One master's commands on one
// channel execute in command order (LiteDRAM: one port, in-order bank queues; the MIG keeps the
// order of accesses to one address): a read issued after a write to its beat returns the written
// data without waiting for n_wdone, and a write counted in n_wdone is visible to the other master
// (XDMA, the host) too.
//
// Per channel, one command a cycle, from (in this priority) the A queue (a read's run of APF
// commands, or a write), the SW queue's next fill read, the SW queue's oldest write, the B queue
// (a read or a write). A command shown and not taken is shown again unchanged (hv); a write shows
// its command and its data together, until both are taken (c_done, d_done). Reads need response
// room, reserved as the command goes: a B read one entry of its channel's RD-beat FIFO, an A run
// APF entries of the AD-beat FIFO; an SW fill read's data goes to its queue slot. Each channel's
// reads return in command order, so a tag FIFO per channel (source, SW slot; its head registered)
// routes every returning beat to B, A or the SW slot. The response FIFOs check for overflow in
// simulation: read data never arrives without reserved room.
//
// Port B: chunk reads and word-masked chunk writes; a partial beat goes out with its byte mask
// (the channel module does the controller's read-modify-write, or the MIG's wr_bytes). Port A:
// single-word reads and byte-enabled word writes (a partial beat). Port SW: byte-enabled word
// writes (the quantizer's QST stores), independent of A so that they never hold up the MXU's
// scale reads. SW writes gather per channel in a one-beat buffer: writes to the same beat merge,
// and the beat goes to the SW queue when a write to another beat arrives, when all its bytes are
// written, or after WGATHER cycles without an SW write (so a QST's byte stream costs one write per
// beat, not per byte). wr_idle stays low while a beat is gathered, so a QST still completes only
// once its writes are in memory. Gathered beats go out in order through one queue per channel; a
// beat with bytes missing (a transposed V column: one byte per beat) first reads its beat (a fill
// read) and is written whole, so the controller's read-modify-write, which holds the channel for a
// read's whole latency, never runs for a QST. The queue holds WQD beats per channel, so that the
// fill reads of a transposed V column overlap (a slot lives from its push until its write goes
// out: about a read latency). The queue's writes wait while its fill reads still go out (unless
// it is half full), so the controller sees runs of reads, then runs of writes.
// A reads that fall in the beat of the previous A read (the MXU's scale stream: 16 scales per
// beat) reuse it without a DRAM access. An A read that misses fetches a run of APF
// channel-consecutive beats (APF read commands); the A reads that follow the run in order take
// its beats without a DRAM access, and a read off the run (or after an A write) drops the beats
// not yet taken.
//
// Ordering. otpu_axi_dram waited for AXI write responses; here the command stream orders instead.
// Once a command has entered a channel's stream, stream order is memory order. Before that, the
// A, SW and B queues may reorder against each other, as AXI's read and write channels did: the
// slice orders what depends across ports (a read that depends on a QST or a B write comes after
// wr_idle). Within a queue, requests go out in order. So:
// - An SW fill read waits until every older SW write of its beat has entered the stream (not
//   until it is done): a counter per bucket (a hash of the beat) counts the slots past the read
//   point whose write has not gone out, and a read waits while its bucket's count is not zero (a
//   collision only delays it). The fill read then follows those writes in the stream and reads
//   their data.
// - A write drops channel c's A run, or the reused beat, when it has gone out on c (command and
//   data taken; the stream carries nothing else meanwhile) and its beat is in them: the run's
//   commands before it read the old beat, the ones after it the new. The drop lands at the end
//   of the cycle after the write goes out. An A read taken before then may still take the old
//   beat; it does not depend on the write: the slice waits for wr_idle, which is high in that
//   cycle at the earliest, and an A request then waits a cycle in the A register (a_v). An A
//   write also drops them when it is taken: the A reads after it in the A queue would otherwise
//   take a beat fetched before it. So the QSTs that stream while an MM runs, and DSTEP's state
//   write-back, cost the MXU's scale stream no runs.
// - wr_idle: every accepted write has gone out and is counted in n_wdone.
// Requests are taken when req && rdy; rdy depends on registered state only. Reads return in
// order per port (the B tag with its data).
module otpu_native_dram #(
  parameter int D = 128,
  parameter int QD = 16,                             // request queue depth per channel and port
  parameter int WQD = 32,                            // SW (gathered beat) queue depth per channel
  parameter bit CHASH = 1'b1,                        // address map: see the top
  parameter int WGATHER = 4,                         // idle cycles before a gathered SW beat goes out
  parameter int RD = 128,                            // B read beats in flight per channel
  parameter int AD = 32,                             // A read beats in flight per channel
  parameter int APF = 8                              // A read run (prefetch), beats
) (
  input  logic              clk,
  input  logic              rst,
  // slice side
  output logic              a_rdy_x,
  input  logic              a_req_x,
  input  logic              a_we_x,
  input  logic [31:0]       a_addr_x,   // word address
  input  logic [31:0]       a_wdata_x,
  input  logic [3:0]        a_be_x,
  output logic              a_rvalid,
  output logic [31:0]       a_rdata,
  output logic [31:0]       a_rdata2,   // the word at a_addr ^ 1 (the other half of its 8-byte pair)
  output logic              sw_rdy,
  input  logic              sw_req,
  input  logic [31:0]       sw_addr,    // word address
  input  logic [31:0]       sw_wdata,
  input  logic [3:0]        sw_be,
  output logic              b_rdy,
  input  logic              b_req,
  input  logic              b_tag,
  input  logic              b_we,
  input  logic [D/4-1:0]    b_wmask,
  input  logic [D*8-1:0]    b_wdata,
  input  logic [31:0]       b_addr,     // word address (chunk aligned)
  input  logic              b_par,      // ^b_addr[31:5] (the channel hash; see b_sw)
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  output logic              wr_idle,
  // native memory masters, one per channel ([1:0] = channel)
  output logic [1:0]        n_cvalid,
  input  logic [1:0]        n_cready,
  output logic [1:0]        n_cwe,
  output logic [1:0][24:0]  n_caddr,    // the beat in the channel (channel address bits 30:6)
  output logic [1:0]        n_wvalid,   // write data: one beat per write command, in order
  input  logic [1:0]        n_wready,
  output logic [1:0][511:0] n_wdata,
  output logic [1:0][63:0]  n_wmask,    // 1 = write the byte
  input  logic [1:0]        n_rvalid,   // read data, in read-command order, no backpressure
  input  logic [1:0][511:0] n_rdata,
  input  logic [1:0][15:0]  n_wdone     // write beats taken by the controller (mod 2^16)
);
  initial if (D != 128) $fatal(1, "otpu_native_dram: D must be 128 (one beat per channel)");
  // (AD >= 2 APF - 1: a run can start while the previous run's untaken beats are stored)
  initial if (APF < 1 || APF > 8 || AD < 2 * APF - 1) $fatal(1, "otpu_native_dram: bad APF");
  localparam int QW = $clog2(QD);
  localparam int WW = $clog2(WQD);
  localparam int NH = 64;                            // SW hazard buckets (see w_blk)
  localparam int HW = $clog2(NH);
  localparam int RW = $clog2(RD);
  localparam int AW_ = $clog2(AD);
  localparam int OD = 2 * RD;                        // B tags / A order entries in flight (2^k)
  localparam int OW = $clog2(OD);
  localparam int TD = 2 ** $clog2(RD + AD + WQD);    // reads in flight per channel (tags)
  localparam int TW = $clog2(TD);
  localparam int PW = $clog2(4 * QD + 2 * WQD + 1);  // writes accepted and not gone out
  localparam logic [1:0] K_B = 2'd0, K_A = 2'd1, K_W = 2'd2;          // a read's source
  localparam logic [1:0] S_A = 2'd0, S_WR = 2'd1, S_WW = 2'd2, S_B = 2'd3;  // a command's

  // the hazard bucket of a channel beat: an XOR fold, so that the strided beats of a transposed
  // V column spread over the buckets
  function automatic logic [HW-1:0] whash(input logic [24:0] a);
    return a[5:0] ^ a[11:6] ^ a[17:12] ^ a[23:18] ^ {5'd0, a[24]};
  endfunction

  // n + inc - dec as a choice among n - 1, n and n + 1: the counts the slice's requests move
  // (the B and SW queues, the B tags) take their late strobes after the adders, not before
  function automatic logic [15:0] updn(input logic [15:0] n, input logic inc, input logic dec);
    return inc == dec ? n : inc ? n + 1'b1 : n - 1'b1;
  endfunction

  // the channel of a word address's beat
  function automatic logic chan_of(input logic [31:0] word_addr);
    return word_addr[4] ^ (CHASH && ^word_addr[31:5]);
  endfunction

  // ------------------------------------------------------------------ request queues
  // per channel: qb (port B beats), qa (port A beats), qw (gathered SW beats); m: the channel beat
  typedef struct packed {
    logic         we;
    logic [24:0]  m;
    logic [511:0] data;
    logic [15:0]  wmask;       // word enables
  } qb_t;
  typedef struct packed {
    logic         we;
    logic [24:0]  m;
    logic [3:0]   idx;         // word in the beat
    logic [31:0]  data;
    logic [3:0]   be;
  } qa_t;
  typedef struct packed {      // a gathered SW beat
    logic [24:0]  m;
    logic [511:0] data;
    logic [63:0]  strb;
  } qw_t;
  typedef struct packed {      // a read in flight: its source, and an SW fill read's slot
    logic [1:0]    k;
    logic [WW-1:0] slot;
  } tg_t;

  logic [QW:0] qb_n [2];
  logic [QW:0] qb_h [2];                    // (2 QD slots: see qbm)
  logic [QW:0] qa_n [2];
  logic [QW-1:0] qa_h [2];
  logic [2:0]  a_iss [2];                   // the head A read's run: beats already issued
  // port SW (gathered beats): live entries (until their write goes out) from qw_f, the next to
  // write; qw_r the next whose fill read is due (qw_rn from qw_r on)
  logic [WW:0] qw_n [2], qw_rn [2];
  logic [WW-1:0] qw_f [2], qw_r [2];
  logic [WQD-1:0] wpart [2], wgot [2];      // per slot: bytes missing; its fill data is in
  // read-after-write hazards: whb counts, per bucket (a hash of the beat), the live entries past
  // the read point (from qw_f up to qw_r: fill read issued, or whole); wnz: whb != 0. A count is
  // read only while its bucket is live (wnz): the reset clears wnz alone (on core_rst's fan-out,
  // the counts' 2 x NH x (WW+1) flip-flops missed 100 MHz on 9.1 ns of route) and an increment
  // of a bucket that is not live starts it at 1
  logic [WW:0]  whb [2][NH];
  logic [NH-1:0] wnz [2];
  logic [1:0]   h_inc, h_dec;               // slot qw_r passes the read point / qw_f's write goes
  // reads in flight per channel, in command order; the oldest one's tag registered (tgh)
  logic [TW:0]  tg_n [2];
  logic [TW-1:0] tg_h [2];
  tg_t          tgh [2], tg_in [2], tg_nx1 [2];
  // SW gather buffers (see the top): valid, beat and bytes written, cycles since the last merge.
  // Their data (gd) takes an SW write a cycle after it is taken (gw_*: taken last cycle, to the
  // buffer's beat), so the take (at the end of QUANT's TMEM grant) enables only gb's 89
  // flip-flops per channel, not the data's 512 (the 5e5a build's widest DMA -> memory group:
  // sleft -> gb data, 2048 endpoints); a gathered beat's data goes into the SW queue a cycle
  // after it is pushed (qwd)
  typedef struct packed {
    logic [24:0]  m;
    logic [63:0]  strb;
  } gs_t;
  logic [1:0]   gv;
  gs_t          gb [2];
  logic [511:0] gd [2];
  logic [2:0]   gage [2];
  logic [1:0]   gw_v, gw_same;
  logic [3:0]   gw_word, gw_be;
  logic [31:0]  gw_wdata;

  // order of B reads (tags) and of A reads (channel, word, reuse)
  logic [1:0]   bt_q [OD];                     // LUT RAM: tag, halves swapped (CHASH)
  logic [OW:0]  bt_n;
  logic [OW-1:0] bt_h;
  // A reads in order: channel, word, reuse of the last beat, run beats to drop before this one
  typedef struct packed { logic c; logic [3:0] idx; logic reuse; logic [2:0] drop; } ao_t;
  ao_t          ao_q [OD];
  ao_t          aoh;                           // the oldest A read
  logic [OW:0]  ao_n;
  logic [OW-1:0] ao_h;

  // A beat reuse: the logical beat (bits 29:4 of the word address) and channel of the last A read
  logic         al_v, al_c;
  logic [25:0]  al_beat;

  assign b_rdy = (qb_n[0] < QD) && (qb_n[1] < QD) && (bt_n < OD);
  // Port A requests are registered on the way in (a_v and the a_* copies): the slice's grant
  // chain (DMA busy -> port arbitration -> the MXU's issue -> the scale address) does not run on
  // into the channel hash, the run compare and the queue writes in one cycle. The register takes
  // a request when it is empty or its request is taken.
  logic        a_v, a_req, a_we, a_rdy;
  logic [31:0] a_addr, a_wdata;
  logic [3:0]  a_be;
  assign a_req = a_v;
  assign a_rdy = (qa_n[0] < QD) && (qa_n[1] < QD) && (ao_n < OD);
  assign a_rdy_x = !a_v || a_rdy;
  always_ff @(posedge clk) begin
    if (rst) a_v <= 1'b0;
    else if (a_req_x && a_rdy_x) a_v <= 1'b1;
    else if (a_rdy) a_v <= 1'b0;
    if (a_req_x && a_rdy_x) begin
      a_we <= a_we_x; a_addr <= a_addr_x; a_wdata <= a_wdata_x; a_be <= a_be_x;
    end
  end
  assign sw_rdy = (qw_n[0] < WQD) && (qw_n[1] < WQD);
  wire sw_take = sw_req && sw_rdy;
  wire sw_ch = chan_of(sw_addr);
  wire [24:0] sw_m = sw_addr[29:5];
  wire b_take = b_req && b_rdy;
  wire a_take = a_req && a_rdy;
  wire a_ch = chan_of(a_addr);
  // the chunk's high half is on channel 0. b_addr comes at the end of the slice's grant chain
  // (DMA request -> the MXU's grant and request -> the port's address mux), and its 27-bit parity
  // then ran into every queue write (b_par: the slice muxes its sources' parities instead)
  wire b_sw = CHASH && b_par;
  wire [1:0] b_hnz = {b_wmask[31:16] != 0, b_wmask[15:0] != 0};   // the halves a B write writes
  wire [25:0] a_beat = a_addr[29:4];
  wire a_reuse = !a_we && al_v && al_beat == a_beat;
  // A runs per channel: the next channel beat of the run, valid (no write since), beats not
  // yet taken (in flight or stored)
  logic [24:0]  pnx [2];
  logic [1:0]   pv;
  logic [2:0]   pfl [2];
  wire  [24:0]  a_cb = a_addr[29:5];              // the channel beat
  wire  a_hit = !a_we && !a_reuse && pv[a_ch] && a_cb == pnx[a_ch] && pfl[a_ch] != 0;

  // ------------------------------------------------------------------ response FIFOs
  logic [RW:0]  rb_n [2], rb_res [2];          // stored; stored + in flight
  logic [RW-1:0] rb_h [2], rb_t [2];
  logic [AW_:0] ra_n [2], ra_res [2];
  logic [AW_-1:0] ra_h [2], ra_t [2];

  // ------------------------------------------------------------------ queue and FIFO memories
  // (per channel; written in their own processes so they map to LUT RAM)
  logic [1:0]   qb_push, qa_push, qw_push;
  qb_t          qb_e [2];
  qa_t          qa_e [2];
  qb_t          hb [2];
  qa_t          ha [2];
  qw_t          hw [2];
  // slot qw_r (the read point): its beat, bucket and whether it is partial, registered (wadr_r,
  // wh_r, wp_r; see their process); the next slot's beat and bucket; the bucket of slot qw_f
  logic [24:0]  wadr_r [2], wadr_n [2];
  logic [HW-1:0] wh_r [2], wh_n [2], wh_f [2];
  logic [1:0]   wp_r;
`ifndef SYNTHESIS
  logic [24:0]  wadr_o [2];
  logic [HW-1:0] wh_o [2];
`endif
  logic [511:0] rb_head [2], ra_head [2], wr_head [2];
  logic [1:0]   rtk;                               // a read command taken (tag push)
  always_comb begin
    for (int c = 0; c < 2; c++) begin
      // channel c holds the chunk's half c ^ b_sw
      qb_push[c] = b_take && (!b_we || b_hnz[c ^ b_sw]);
      qb_e[c].we = b_we;
      qb_e[c].m = b_addr[29:5];
      qb_e[c].data = b_wdata[512 * (c ^ b_sw) +: 512];
      qb_e[c].wmask = b_we ? b_wmask[16 * (c ^ b_sw) +: 16] : '0;
      qa_push[c] = a_take && a_ch == c[0] && !a_reuse && !a_hit;
      qa_e[c].we = a_we;
      qa_e[c].m = a_cb;
      qa_e[c].idx = a_addr[3:0];
      qa_e[c].data = a_wdata;
      qa_e[c].be = a_be;
      // the gathered beat goes out: another beat's SW write (room is sure: sw_rdy), or, with
      // no SW write this cycle and room in the queue, full or idle
      qw_push[c] = gv[c] && (sw_take && sw_ch == c[0] ? gb[c].m != sw_m
                   : qw_n[c] < WQD && (&gb[c].strb || gage[c] >= 3'(WGATHER)));
    end
  end
  // the SW gather buffers' data: last cycle's write merged (see gd); last cycle's tail slot
  logic [WW-1:0] ql_t [2];
  logic [1:0]    ql_v;
  always_ff @(posedge clk) begin
    for (int c = 0; c < 2; c++) begin
      ql_t[c] <= WW'(qw_f[c] + qw_n[c]);
      ql_v[c] <= !qw_n[c][WW];
      gw_same[c] <= gv[c] && gb[c].m == sw_m;
      if (gw_v[c])
        for (int k = 0; k < 64; k++)
          gd[c][8 * k +: 8] <= gw_word == 4'(k / 4) && gw_be[k % 4] ? gw_wdata[8 * (k % 4) +: 8]
                               : gw_same[c] ? gd[c][8 * k +: 8] : 8'h00;
    end
    gw_word <= sw_addr[3:0]; gw_wdata <= sw_wdata; gw_be <= sw_be;
  end

  for (genvar c = 0; c < 2; c++) begin : g_mem
    // flat vectors: Vivado builds a RAM of structs from registers
    // qbm: 2 QD slots for at most QD entries (b_rdy), so the slot at the tail is never live and
    // is written every cycle; a push only moves the tail. Its write enable was the push, which
    // comes at the end of the slice's grant chain and fanned out to all of the queue's LUT RAM
    // (qb_t's 554 bits: 745 cells per channel; the 959b425 build's worst DMA -> memory paths,
    // -0.390 ns at 133.33 MHz). At QD = 16 the RAM32Ms hold 32 slots already
    logic [$bits(qb_t)-1:0] qbm [2 * QD];
    logic [$bits(qa_t)-1:0] qam [QD];
    logic [$bits(gs_t)-1:0] qwm [WQD];      // per slot: the beat and its bytes written
    logic [511:0] qwd [WQD];                // and their data (see gd)
    logic [24:0] wam [WQD];                 // per slot: the beat
    logic [HW-1:0] whm [WQD];               // per slot: its hazard bucket
    logic [$bits(tg_t)-1:0] tgm [TD];       // the reads in flight
    logic [511:0] rbm [RD];
    logic [511:0] ram [AD];
    logic [511:0] wrm [WQD];                // the SW queue's fill data, per slot
    always_ff @(posedge clk) qbm[(QW + 1)'(qb_h[c] + qb_n[c])] <= qb_e[c];
    always_ff @(posedge clk) if (qa_push[c]) qam[QW'(qa_h[c] + qa_n[c])] <= qa_e[c];
    // the SW queue's tail slot (not live unless the queue is full) takes the gathered beat every
    // cycle: its write enables are registers, not the push (an SW write's take comes at the end
    // of QUANT's TMEM grant; the 5e5a build's worst DMA -> memory paths ran into qwm's WE). The
    // data goes into last cycle's tail slot (ql_t, ql_v) a cycle later, as gd has it: a slot
    // pushed last cycle is at the read point or before it, and its data is read (hw) only past it
    always_ff @(posedge clk) if (!qw_n[c][WW]) qwm[WW'(qw_f[c] + qw_n[c])] <= gb[c];
    always_ff @(posedge clk) if (ql_v[c]) qwd[ql_t[c]] <= gd[c];
    always_ff @(posedge clk) if (!qw_n[c][WW]) wam[WW'(qw_f[c] + qw_n[c])] <= gb[c].m;
    always_ff @(posedge clk) if (!qw_n[c][WW]) whm[WW'(qw_f[c] + qw_n[c])] <= whash(gb[c].m);
    always_ff @(posedge clk) if (rtk[c]) tgm[TW'(tg_h[c] + tg_n[c])] <= tg_in[c];
    assign tg_nx1[c] = tg_t'(tgm[TW'(tg_h[c] + 1'b1)]);
    assign wadr_n[c] = wam[WW'(qw_r[c] + 1'b1)];
    assign wh_n[c] = whm[WW'(qw_r[c] + 1'b1)];
`ifndef SYNTHESIS
    assign wadr_o[c] = wam[qw_r[c]];                // (the registers' reference)
    assign wh_o[c] = whm[qw_r[c]];
`endif
    assign wh_f[c] = whm[qw_f[c]];
    always_ff @(posedge clk)
      if (n_rvalid[c] && tgh[c].k == K_B) rbm[rb_t[c]] <= n_rdata[c];
    always_ff @(posedge clk)
      if (n_rvalid[c] && tgh[c].k == K_A) ram[ra_t[c]] <= n_rdata[c];
    always_ff @(posedge clk)
      if (n_rvalid[c] && tgh[c].k == K_W) wrm[tgh[c].slot] <= n_rdata[c];
    assign wr_head[c] = wrm[qw_f[c]];
    assign hb[c] = qb_t'(qbm[qb_h[c]]);
    assign ha[c] = qa_t'(qam[qa_h[c]]);
    gs_t hs;
    assign hs = gs_t'(qwm[qw_f[c]]);
    assign hw[c] = qw_t'{m: hs.m, data: qwd[qw_f[c]], strb: hs.strb};
    assign rb_head[c] = rbm[rb_h[c]];
    assign ra_head[c] = ram[AW_'(ra_h[c] + AW_'(aoh.drop))];
  end

  // ------------------------------------------------------------------ per-channel issue
  logic [1:0] hv;                                // a command shown and not taken (or a write
  logic [1:0] hsrc [2];                          //   not yet out): shown again; its source
  logic [1:0] c_done, d_done;                    // the shown write's command / data taken
  logic [1:0] sv;                                // a command is shown
  logic [1:0] src [2];                           // its source (S_*)
  logic [1:0] swr;                               // it is a write
  logic [1:0] ctk, dtk;                          // command / write data taken this cycle
  logic [1:0] wiss;                              // a write goes out (both taken)
  logic [1:0] w_blk, w_hold;                     // the fill read waits (an older write of its
                                                 //   beat); the SW writes wait
  always_comb begin
    for (int c = 0; c < 2; c++) begin
      logic e_a, e_wr, e_ww, e_b;
      // the SW queue's next partial beat: blocked while an older live entry may have its
      // address (one in its bucket; a hash collision only delays the read)
      w_blk[c] = wnz[c][wh_r[c]];
      // the queue's writes wait while its fill reads still go out (not blocked), unless half full
      w_hold[c] = (qw_rn[c] != 0) && wp_r[c] && !w_blk[c] && qw_n[c] < (WW + 1)'(WQD / 2);
      // A: a read's run (response room for the whole run as it starts), or a write
      e_a = (qa_n[c] != 0) && (ha[c].we || a_iss[c] != 0 || ra_res[c] <= (AW_ + 1)'(AD - APF));
      e_wr = (qw_rn[c] != 0) && wp_r[c] && !w_blk[c];
      // an SW write once its slot is past the read point and, if partial, has its fill data
      e_ww = (qw_n[c] != qw_rn[c]) && (!wpart[c][qw_f[c]] || wgot[c][qw_f[c]]) && !w_hold[c];
      e_b = (qb_n[c] != 0) && (hb[c].we || rb_res[c] < (RW + 1)'(RD));
      sv[c] = hv[c] || e_a || e_wr || e_ww || e_b;
      src[c] = hv[c] ? hsrc[c] : e_a ? S_A : e_wr ? S_WR : e_ww ? S_WW : S_B;
      swr[c] = (src[c] == S_A && ha[c].we) || src[c] == S_WW || (src[c] == S_B && hb[c].we);
      n_cvalid[c] = sv[c] && !(swr[c] && c_done[c]);
      n_wvalid[c] = sv[c] && swr[c] && !d_done[c];
      n_cwe[c] = swr[c];
      case (src[c])
        S_A:  n_caddr[c] = ha[c].m + 25'(a_iss[c]);
        S_WR: n_caddr[c] = wadr_r[c];
        S_WW: n_caddr[c] = hw[c].m;
        default: n_caddr[c] = hb[c].m;
      endcase
      n_wdata[c] = '0;
      n_wmask[c] = '0;
      if (src[c] == S_WW) begin
        for (int k = 0; k < 64; k++)
          n_wdata[c][8 * k +: 8] = hw[c].strb[k] ? hw[c].data[8 * k +: 8] : wr_head[c][8 * k +: 8];
        n_wmask[c] = '1;
      end else if (src[c] == S_A) begin
        n_wdata[c][32 * ha[c].idx +: 32] = ha[c].data;
        n_wmask[c][4 * ha[c].idx +: 4] = ha[c].be;
      end else begin
        n_wdata[c] = hb[c].data;
        for (int k = 0; k < 16; k++) n_wmask[c][4 * k +: 4] = {4{hb[c].wmask[k]}};
      end
      ctk[c] = n_cvalid[c] && n_cready[c];
      dtk[c] = n_wvalid[c] && n_wready[c];
      wiss[c] = sv[c] && swr[c] && (c_done[c] || ctk[c]) && (d_done[c] || dtk[c]);
      rtk[c] = ctk[c] && !swr[c];
      tg_in[c].k = src[c] == S_A ? K_A : src[c] == S_WR ? K_W : K_B;
      tg_in[c].slot = qw_r[c];
    end
  end

  // a write that went out on channel c (last cycle: wk_v, its beat wk_m) drops c's A run, or the
  // reused beat, when its beat is in them (see the top); a cycle late, off the path from the
  // command select and the channel's readies, and still before any A read that depends on the
  // write is taken (that comes after wr_idle, and waits a cycle in the A register)
  logic [1:0] wk_v, wk_r, wk_a;
  logic [24:0] wk_m [2];
  always_ff @(posedge clk) begin
    wk_v <= rst ? '0 : wiss;
    for (int c = 0; c < 2; c++) wk_m[c] <= n_caddr[c];
  end
  always_comb
    for (int c = 0; c < 2; c++) begin
      wk_r[c] = wk_v[c] && pv[c] && 25'(wk_m[c] - pnx[c]) < 25'(pfl[c]);
      wk_a[c] = wk_v[c] && al_v && al_c == c[0] && al_beat[25:1] == wk_m[c];
    end

  // ------------------------------------------------------------------ merge
  wire b_out = (bt_n != 0) && (rb_n[0] != 0) && (rb_n[1] != 0);
  // the oldest A read, registered (the order FIFO's LUT RAM read stays out of the run-beat and
  // word select into a_rdata): after a pop the next entry (written at least a cycle before), or
  // the entry pushed into an empty queue
  ao_t          ao_nx1, ao_in;
  assign ao_nx1 = ao_q[OW'(ao_h + 1'b1)];
  assign ao_in = '{c: a_ch, idx: a_addr[3:0], reuse: a_reuse,
                   drop: (a_reuse || a_hit) ? 3'd0 : pfl[a_ch]};
  logic [511:0] a_last;                          // the beat of the last fetched A read
  wire a_out = (ao_n != 0) && (aoh.reuse || ra_n[aoh.c] > (AW_ + 1)'(aoh.drop));
  wire [511:0] a_src = aoh.reuse ? a_last : ra_head[aoh.c];
  always_ff @(posedge clk)
    if (a_out) aoh <= (ao_n == (OW + 1)'(1)) ? ao_in : ao_nx1;
    else if (ao_n == 0) aoh <= ao_in;
  // the oldest read in flight's tag, registered likewise (it steers the response FIFO writes)
  always_ff @(posedge clk)
    for (int c = 0; c < 2; c++)
      if (n_rvalid[c]) tgh[c] <= (tg_n[c] == (TW + 1)'(1)) ? tg_in[c] : tg_nx1[c];
      else if (tg_n[c] == 0) tgh[c] <= tg_in[c];
  assign b_rvalid = b_out;
  wire [1:0] bth = bt_q[bt_h];
  assign b_rtag = bth[0];
  assign b_rdata = bth[1] ? {rb_head[0], rb_head[1]} : {rb_head[1], rb_head[0]};
  // the A read data registered on the way out (the head's run-beat and word select ran into the
  // MXU's scale FIFO block RAM in one cycle)
  always_ff @(posedge clk) begin
    a_rvalid <= !rst && a_out;
    a_rdata <= a_src[32 * aoh.idx +: 32];
    a_rdata2 <= a_src[32 * (aoh.idx ^ 4'd1) +: 32];
  end

  // the tag FIFO writes its tail every cycle too (the slot is not live unless the FIFO is full),
  // so its write enable is a register, not b_take
  always_ff @(posedge clk) begin
    if (!bt_n[OW]) bt_q[OW'(bt_h + bt_n)] <= {b_sw, b_tag};
    if (a_take && !a_we)
      ao_q[OW'(ao_h + ao_n)] <= ao_in;
  end

  // ------------------------------------------------------------------ writes outstanding
  // wq_n counts the writes (beats) accepted before the last cycle that have not gone out; wacc_q
  // holds the last cycle's accepts (folded in a cycle late, off the rdy -> take path). A write
  // goes out at least a cycle after its accept, so wq_n never goes negative. iss_w[c] counts
  // channel c's writes gone out (mod 2^16), which n_wdone reaches once the controller has them;
  // it starts at n_wdone's value in reset (the channel module's count need not reset with rst,
  // as long as no write of ours is still on its way to the controller then).
  logic [PW-1:0] wq_n;
  logic [4:0]    wacc_q;
  logic [15:0]   iss_w [2];
  assign wr_idle = (wq_n == 0) && (wacc_q == '0) && (gv == '0) && !(a_v && a_we) &&
                   (iss_w[0] == n_wdone[0]) && (iss_w[1] == n_wdone[1]);

  always_ff @(posedge clk) begin
    if (rst) begin
      for (int c = 0; c < 2; c++) begin
        qb_n[c] <= '0; qb_h[c] <= '0; qa_n[c] <= '0; qa_h[c] <= '0; a_iss[c] <= '0;
        qw_n[c] <= '0; qw_f[c] <= '0; qw_r[c] <= '0; qw_rn[c] <= '0; wnz[c] <= '0;
        tg_n[c] <= '0; tg_h[c] <= '0;
        rb_n[c] <= '0; rb_res[c] <= '0; rb_h[c] <= '0; rb_t[c] <= '0;
        ra_n[c] <= '0; ra_res[c] <= '0; ra_h[c] <= '0; ra_t[c] <= '0;
        iss_w[c] <= n_wdone[c];
      end
      bt_n <= '0; bt_h <= '0; ao_n <= '0; ao_h <= '0;
      al_v <= 1'b0;
      pv <= '0; pfl[0] <= '0; pfl[1] <= '0;
      wq_n <= '0; wacc_q <= '0;
      gv <= '0; gage[0] <= '0; gage[1] <= '0; gw_v <= '0;
      hv <= '0; c_done <= '0; d_done <= '0;
    end else begin
      // writes accepted (the take strobes: an A or an SW write goes to exactly one channel, so
      // a_reuse and the channel decode stay off this path; a B write to each channel whose half
      // it writes) and gone out
      wq_n <= wq_n + PW'(wacc_q[0]) + PW'(wacc_q[1]) + PW'(wacc_q[2]) + PW'(wacc_q[3]) +
              PW'(wacc_q[4]) - PW'(wiss[0]) - PW'(wiss[1]);
      wacc_q <= {qw_push[1], qw_push[0], a_take && a_we,
                 b_take && b_we && b_wmask[31:16] != 0,
                 b_take && b_we && b_wmask[15:0]  != 0};
      for (int c = 0; c < 2; c++) begin
        logic [QW:0] na;
        logic [RW:0] rbn, rbr;
        logic [AW_:0] ran, rar;
        logic [TW:0] ntg;
        logic popb, popa, popw;
        na = qa_n[c]; ntg = tg_n[c];
        rbn = rb_n[c]; rbr = rb_res[c]; ran = ra_n[c]; rar = ra_res[c];
        // ---- accept
        if (qa_push[c]) na = na + 1;
        if (!qw_n[c][WW]) begin                // the tail slot, as qwm (see g_mem)
          logic [WW-1:0] t;
          t = WW'(qw_f[c] + qw_n[c]);
          wpart[c][t] <= !(&gb[c].strb);
          wgot[c][t] <= 1'b0;
        end
        // ---- the command stream: a shown command is shown again until taken, a write until
        // its command and its data are taken
        hv[c] <= sv[c] && (swr[c] ? !wiss[c] : !ctk[c]);
        hsrc[c] <= src[c];
        c_done[c] <= swr[c] && !wiss[c] && (c_done[c] || ctk[c]);
        d_done[c] <= swr[c] && !wiss[c] && (d_done[c] || dtk[c]);
        popa = src[c] == S_A && (swr[c] ? wiss[c] : ctk[c] && a_iss[c] == 3'(APF - 1));
        popb = src[c] == S_B && (swr[c] ? wiss[c] : ctk[c]);
        popw = src[c] == S_WW && wiss[c];
        if (rtk[c]) ntg = ntg + 1;
        if (src[c] == S_A && rtk[c]) begin      // an A run's beat; the run reserves its room
          a_iss[c] <= (a_iss[c] == 3'(APF - 1)) ? 3'd0 : a_iss[c] + 1'b1;
          if (a_iss[c] == 0) rar = rar + (AW_ + 1)'(APF);
        end
        if (src[c] == S_B && rtk[c]) rbr = rbr + 1;
        // a fill read passes the read point; a whole beat needs no fill read (e_wr is never set
        // for one): it passes the read point at once, so before its write can go (h_inc: either)
        if (h_inc[c]) qw_r[c] <= qw_r[c] + 1;
        if (popb) qb_h[c] <= qb_h[c] + 1;
        if (popa) begin qa_h[c] <= qa_h[c] + 1; na = na - 1; end
        if (popw) qw_f[c] <= qw_f[c] + 1;
        if (wiss[c]) iss_w[c] <= iss_w[c] + 1'b1;
        // ---- read data, routed by the oldest read's tag
        if (n_rvalid[c]) begin
          case (tgh[c].k)
            K_B: begin rb_t[c] <= rb_t[c] + 1; rbn = rbn + 1; end
            K_A: begin ra_t[c] <= ra_t[c] + 1; ran = ran + 1; end
            default: wgot[c][tgh[c].slot] <= 1'b1;   // the SW fill read: its slot has the data
          endcase
          tg_h[c] <= tg_h[c] + 1;
          ntg = ntg - 1;
        end
        // ---- merge pops
        if (b_out) begin
          rb_h[c] <= rb_h[c] + 1;
          rbn = rbn - 1; rbr = rbr - 1;
        end
        if (a_out && !aoh.reuse && aoh.c == c[0]) begin    // (and the dropped run beats)
          ra_h[c] <= ra_h[c] + AW_'(aoh.drop) + 1'b1;
          ran = ran - (AW_ + 1)'(aoh.drop) - 1'b1;
          rar = rar - (AW_ + 1)'(aoh.drop) - 1'b1;
        end
        // ---- hazard buckets: slot qw_r passes the read point, slot qw_f's write goes out
        for (int k = 0; k < NH; k++)
          if (h_ik[c][k] || h_dk[c][k]) wnz[c][k] <= h_iv[c][k] || whb[c][wh_f[c]] != (WW + 1)'(1);
        qb_n[c] <= (QW + 1)'(updn(16'(qb_n[c]), qb_push[c], popb));
        qw_n[c] <= (WW + 1)'(updn(16'(qw_n[c]), qw_push[c], popw));
        qw_rn[c] <= (WW + 1)'(updn(16'(qw_rn[c]), qw_push[c], h_inc[c]));
        qa_n[c] <= na; tg_n[c] <= ntg;
        rb_n[c] <= rbn; rb_res[c] <= rbr; ra_n[c] <= ran; ra_res[c] <= rar;
      end
      // ---- order FIFOs
      begin
        logic [OW:0] aon;
        aon = ao_n;
        if (b_out) bt_h <= bt_h + 1;
        if (a_take && !a_we) aon = aon + 1;
        if (a_out) begin
          ao_h <= ao_h + 1; aon = aon - 1;
          if (!aoh.reuse) a_last <= ra_head[aoh.c];
        end
        bt_n <= (OW + 1)'(updn(16'(bt_n), b_take && !b_we, b_out));
        ao_n <= aon;
      end
      // ---- SW gather: merge into the beat, or start a new one (the old one was pushed)
      for (int c = 0; c < 2; c++) begin
        gw_v[c] <= sw_take && sw_ch == c[0];
        if (sw_take && sw_ch == c[0]) begin
          logic same;
          same = gv[c] && gb[c].m == sw_m;
          gv[c] <= 1'b1;
          gage[c] <= '0;
          gb[c].m <= sw_m;
          for (int k = 0; k < 64; k++)
            gb[c].strb[k] <= (sw_addr[3:0] == 4'(k / 4) && sw_be[k % 4]) || (same && gb[c].strb[k]);
        end else if (qw_push[c]) begin
          gv[c] <= 1'b0;
        end else if (gv[c] && gage[c] != '1) begin
          gage[c] <= gage[c] + 1;
        end
      end
      // ---- A runs: a hit takes the next beat, a miss starts a run (dropping the rest)
      if (a_take && !a_we && !a_reuse) begin
        if (a_hit) begin
          pnx[a_ch] <= pnx[a_ch] + 1;
          pfl[a_ch] <= pfl[a_ch] - 1;
        end else begin
          pnx[a_ch] <= a_cb + 1;
          pfl[a_ch] <= 3'(APF - 1);
          pv[a_ch] <= 1'b1;
        end
      end
      for (int c = 0; c < 2; c++) if (wk_r[c]) pv[c] <= 1'b0;
      if (a_take && a_we) pv <= '0;
      // ---- A beat reuse: the beat of the last A read, forgotten on an A write or a write to it
      if ((a_take && a_we) || (|wk_a)) al_v <= 1'b0;
      else if (a_take && !a_we) begin
        al_v <= 1'b1;
        al_beat <= a_beat;
        al_c <= a_ch;
      end
    end
  end

  // the hazard buckets' counts (no reset: see whb). Per bucket k: incremented (h_ik: slot qw_r,
  // bucket wh_r, passes the read point) or decremented (h_dk: slot qw_f's write, bucket wh_f,
  // goes out), neither when both hit it; each a function of the late h_inc / h_dec and of the
  // bucket decodes, and an increment's value where the bucket is wh_r and h_inc is set (h_iv)
  logic [NH-1:0] h_ik [2], h_dk [2], h_iv [2];
  always_comb
    for (int c = 0; c < 2; c++) begin
      h_inc[c] = (src[c] == S_WR && ctk[c]) || (qw_rn[c] != 0 && !wp_r[c]);
      h_dec[c] = src[c] == S_WW && wiss[c];
      for (int k = 0; k < NH; k++) begin
        h_ik[c][k] = h_inc[c] && wh_r[c] == HW'(k) && !(h_dec[c] && wh_f[c] == wh_r[c]);
        h_dk[c][k] = h_dec[c] && wh_f[c] == HW'(k) && !(h_inc[c] && wh_f[c] == wh_r[c]);
        h_iv[c][k] = h_inc[c] && wh_r[c] == HW'(k);
      end
    end
  // slot qw_r's beat, bucket and partial flag, registered: read from the LUT RAMs and wpart by
  // qw_r, they started the fill read's hazard check (w_blk) and so the command select and the
  // hazard count updates (the 5e5a build's nmem > nmem paths: qw_r -> whm -> wnz -> ... -> whb).
  // qw_r moves by one exactly on h_inc; the slot it then points to is the tail, whose write
  // (see g_mem) lands in it this cycle, when no entry past the read point is left (qw_rn 1, or 0
  // while it stays), otherwise the next slot as stored
  always_ff @(posedge clk)
    for (int c = 0; c < 2; c++) begin
      logic tw;
      tw = !qw_n[c][WW] && qw_rn[c] == (h_inc[c] ? (WW + 1)'(1) : (WW + 1)'(0));
      if (tw) begin
        wadr_r[c] <= gb[c].m; wh_r[c] <= whash(gb[c].m); wp_r[c] <= !(&gb[c].strb);
      end else if (h_inc[c]) begin
        wadr_r[c] <= wadr_n[c]; wh_r[c] <= wh_n[c]; wp_r[c] <= wpart[c][WW'(qw_r[c] + 1'b1)];
      end
    end
  always_ff @(posedge clk)
    for (int c = 0; c < 2; c++)
      for (int k = 0; k < NH; k++)
        if (h_ik[c][k] || h_dk[c][k])
          whb[c][k] <= !h_iv[c][k] ? whb[c][wh_f[c]] - 1'b1 :
                       wnz[c][wh_r[c]] ? whb[c][wh_r[c]] + 1'b1 : (WW + 1)'(1);

`ifndef SYNTHESIS
  // The hazard counts are read only while live: a write goes out only from a live bucket, and
  // every live count is what a count reset with the rest would hold (whb_ref). The counts start
  // with anything, as after a reset in the middle of a run
  logic [WW:0] whb_ref [2][NH];
  initial for (int c = 0; c < 2; c++) for (int k = 0; k < NH; k++) whb[c][k] = (WW + 1)'($urandom);
  always_ff @(posedge clk)
    if (rst) begin
      for (int c = 0; c < 2; c++) for (int k = 0; k < NH; k++) whb_ref[c][k] <= '0;
    end else
      for (int c = 0; c < 2; c++) begin
        if (h_dec[c] && !wnz[c][wh_f[c]])
          $fatal(1, "otpu_native_dram: channel %0d bucket %0d written out while not live", c, wh_f[c]);
        for (int k = 0; k < NH; k++)
          if (wnz[c][k] != (whb_ref[c][k] != 0) || (wnz[c][k] && whb[c][k] != whb_ref[c][k]))
            $fatal(1, "otpu_native_dram: channel %0d bucket %0d count %0d (live %0d), expected %0d",
                   c, k, whb[c][k], wnz[c][k], whb_ref[c][k]);
        if (h_inc[c] && !(h_dec[c] && wh_f[c] == wh_r[c]))
          whb_ref[c][wh_r[c]] <= whb_ref[c][wh_r[c]] + 1'b1;
        if (h_dec[c] && !(h_inc[c] && wh_f[c] == wh_r[c]))
          whb_ref[c][wh_f[c]] <= whb_ref[c][wh_f[c]] - 1'b1;
      end

  // b_par is the parity of the address. The B queues and the tag FIFO write their tails every
  // cycle: the head of each (the entry a command or a read's data takes) is what a queue written
  // on a push only holds (qbr, btr: the old write enables)
  always_ff @(posedge clk)
    if (!rst && b_req && b_par != ^b_addr[31:5])
      $fatal(1, "otpu_native_dram: b_par %0d, address %h", b_par, b_addr);
  logic [$bits(qb_t)-1:0] qbr [2][QD];
  logic [1:0] btr [OD];
  always_ff @(posedge clk) begin
    for (int c = 0; c < 2; c++) if (qb_push[c]) qbr[c][QW'(qb_h[c] + qb_n[c])] <= qb_e[c];
    if (b_take && !b_we) btr[OW'(bt_h + bt_n)] <= {b_sw, b_tag};
  end
  always_ff @(posedge clk)
    if (!rst) begin
      if (qb_n[0] != 0 && hb[0] != qb_t'(qbr[0][QW'(qb_h[0])]))
        $fatal(1, "otpu_native_dram: channel 0 B queue head (slot %0d) differs", qb_h[0]);
      if (qb_n[1] != 0 && hb[1] != qb_t'(qbr[1][QW'(qb_h[1])]))
        $fatal(1, "otpu_native_dram: channel 1 B queue head (slot %0d) differs", qb_h[1]);
      if (bt_n != 0 && bth != btr[bt_h])
        $fatal(1, "otpu_native_dram: B tag %0d at %0d, expected %0d", bth, bt_h, btr[bt_h]);
    end

  // gb and, a cycle later, gd are, while valid, what a buffer merging each write as it is taken
  // holds (gb_ref)
  qw_t gb_ref [2];
  logic [511:0] gd_ref [2];
  logic [1:0] gd_rv;
  always_ff @(posedge clk)
    for (int c = 0; c < 2; c++) begin
      if (!rst && gv[c] && (gb[c].m != gb_ref[c].m || gb[c].strb != gb_ref[c].strb))
        $fatal(1, "otpu_native_dram: channel %0d SW gather buffer %h differs from %h", c, gb[c].m,
               gb_ref[c].m);
      if (!rst && gd_rv[c] && gd[c] != gd_ref[c])
        $fatal(1, "otpu_native_dram: channel %0d SW gather data differs", c);
      gd_ref[c] <= gb_ref[c].data;
      gd_rv[c] <= gv[c];
      if (sw_take && sw_ch == c[0]) begin
        logic same;
        same = gv[c] && gb_ref[c].m == sw_m;
        gb_ref[c].m <= sw_m;
        for (int k = 0; k < 64; k++) begin
          logic hit;
          hit = sw_addr[3:0] == 4'(k / 4) && sw_be[k % 4];
          gb_ref[c].data[8 * k +: 8] <= hit ? sw_wdata[8 * (k % 4) +: 8]
                                        : same ? gb_ref[c].data[8 * k +: 8] : 8'h00;
          gb_ref[c].strb[k] <= hit || (same && gb_ref[c].strb[k]);
        end
      end
    end

  // The SW queue's slots are written at the tail every cycle as well: its slots at the read point
  // (qw_r) and at the head (qw_f) are what a queue written on a push only holds (the head's beat,
  // once past the read point: see qwd); the read point's registers (wadr_r, wh_r, wp_r) are what
  // its slot holds
  logic [$bits(qw_t)-1:0] qwr [2][WQD];
  logic [24:0] war [2][WQD];
  logic [WQD-1:0] wpr [2], wgr [2];
  always_ff @(posedge clk)
    for (int c = 0; c < 2; c++) begin
      if (qw_push[c]) begin
        qwr[c][WW'(qw_f[c] + qw_n[c])] <= gb_ref[c];
        war[c][WW'(qw_f[c] + qw_n[c])] <= gb[c].m;
        wpr[c][WW'(qw_f[c] + qw_n[c])] <= !(&gb[c].strb);
        wgr[c][WW'(qw_f[c] + qw_n[c])] <= 1'b0;
      end
      if (n_rvalid[c] && tgh[c].k == K_W) wgr[c][tgh[c].slot] <= 1'b1;
    end
  always_ff @(posedge clk)
    if (!rst)
      for (int c = 0; c < 2; c++) begin
        if ((src[c] == S_WR && ctk[c]) && (qw_rn[c] != 0 && !wp_r[c]))
          $fatal(1, "otpu_native_dram: channel %0d fill read of a whole beat", c);
        if (qw_rn[c] != 0 && (wadr_r[c] != wadr_o[c] || wh_r[c] != wh_o[c] ||
                              wp_r[c] != wpart[c][qw_r[c]]))
          $fatal(1, "otpu_native_dram: channel %0d read point slot %0d: registered %h %0d %0d, stored %h %0d %0d",
                 c, qw_r[c], wadr_r[c], wh_r[c], wp_r[c], wadr_o[c], wh_o[c], wpart[c][qw_r[c]]);
        if (qw_rn[c] != 0 && (wadr_r[c] != war[c][qw_r[c]] || wh_r[c] != whash(war[c][qw_r[c]]) ||
                              wp_r[c] != wpr[c][qw_r[c]]))
          $fatal(1, "otpu_native_dram: channel %0d SW slot %0d at the read point differs", c, qw_r[c]);
        if (qw_n[c] != 0 && (wh_f[c] != whash(war[c][qw_f[c]]) || wpart[c][qw_f[c]] != wpr[c][qw_f[c]] ||
                             wgot[c][qw_f[c]] != wgr[c][qw_f[c]]))
          $fatal(1, "otpu_native_dram: channel %0d SW head slot %0d differs", c, qw_f[c]);
        if (qw_n[c] != qw_rn[c] && hw[c] != qw_t'(qwr[c][qw_f[c]]))
          $fatal(1, "otpu_native_dram: channel %0d SW head slot %0d's beat differs", c, qw_f[c]);
      end

  // Read data never arrives without a read in flight or without room reserved for it (the
  // native port has no backpressure on read data)
  always_ff @(posedge clk)
    if (!rst)
      for (int c = 0; c < 2; c++) begin
        if (n_rvalid[c] && tg_n[c] == 0)
          $fatal(1, "otpu_native_dram: channel %0d read data with no read in flight", c);
        if (n_rvalid[c] && tg_n[c] != 0 && tgh[c].k == K_B && rb_n[c] == (RW + 1)'(RD))
          $fatal(1, "otpu_native_dram: channel %0d B read FIFO overflow", c);
        if (n_rvalid[c] && tg_n[c] != 0 && tgh[c].k == K_A && ra_n[c] == (AW_ + 1)'(AD))
          $fatal(1, "otpu_native_dram: channel %0d A read FIFO overflow", c);
        if (n_rvalid[c] && tg_n[c] != 0 && tgh[c].k == K_W && wgot[c][tgh[c].slot])
          $fatal(1, "otpu_native_dram: channel %0d SW fill data for a filled slot", c);
        if (rtk[c] && tg_n[c] == (TW + 1)'(TD) && !n_rvalid[c])
          $fatal(1, "otpu_native_dram: channel %0d read tag FIFO overflow", c);
      end

  // Counters for the simulation's statistics (otpu_top prints them): per channel, A runs and
  // their read commands, B reads, SW fill reads; writes by source, and those with a byte mask
  // that is not whole (a read-modify-write in the channel module)
  longint st_arun [2], st_ard [2], st_brd [2], st_srd [2];
  longint st_bwr [2], st_awr [2], st_swr [2], st_pb [2], st_pa [2], st_ps [2];
  initial
    for (int c = 0; c < 2; c++) begin
      st_arun[c] = 0; st_ard[c] = 0; st_brd[c] = 0; st_srd[c] = 0; st_bwr[c] = 0;
      st_awr[c] = 0; st_swr[c] = 0; st_pb[c] = 0; st_pa[c] = 0; st_ps[c] = 0;
    end
  always_ff @(posedge clk)
    for (int c = 0; c < 2; c++) begin
      if (rtk[c] && src[c] == S_A) begin
        st_ard[c] <= st_ard[c] + 1;
        if (a_iss[c] == 0) st_arun[c] <= st_arun[c] + 1;
      end
      if (rtk[c] && src[c] == S_B) st_brd[c] <= st_brd[c] + 1;
      if (rtk[c] && src[c] == S_WR) st_srd[c] <= st_srd[c] + 1;
      if (wiss[c]) begin
        logic part;
        part = n_wmask[c] != '1;
        case (src[c])
          S_A: begin st_awr[c] <= st_awr[c] + 1; if (part) st_pa[c] <= st_pa[c] + 1; end
          S_WW: begin st_swr[c] <= st_swr[c] + 1; if (part) st_ps[c] <= st_ps[c] + 1; end
          default: begin st_bwr[c] <= st_bwr[c] + 1; if (part) st_pb[c] <= st_pb[c] + 1; end
        endcase
      end
    end
`endif
endmodule
