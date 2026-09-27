// Slice DRAM ports (as otpu_dram) on two AXI4 memory channels (the board's two DDR3
// controllers), 512-bit data: 64-byte beats, port B reads in INCR bursts of up to BL beats.
//
// Address map: the slice's byte address space is interleaved over the channels in 64-byte
// beats: logical beat b = addr / 64 lives on channel b % 2 at BASE[b % 2] + (b / 2) * 64. A
// D = 128-byte chunk (port B; requests are chunk aligned) is therefore one beat on each
// channel, and a streamed operand uses both channels evenly. The host applies the same map
// when it fills and reads the memory (opentpu/host/board.py).
//
// Port B: chunk reads and word-masked chunk writes. Port A: single-word reads and byte-enabled
// word writes. Port SW: byte-enabled word writes (the quantizer's QST stores), independent of A
// so that they never hold up the MXU's scale reads. SW writes gather per channel in a one-beat
// buffer: writes to the same beat merge, and the beat goes out as one AXI write when a write to
// another beat arrives, when all its bytes are written, or after WGATHER cycles without an SW
// write (the board's controller does a partial-strobe write as a read-modify-write for ECC, so a
// QST's byte stream costs one write per beat, not per byte). wr_idle stays low while a beat is
// gathered, so a QST still completes only once its writes are in memory. Gathered beats go out
// in order through one queue per channel; a beat with bytes missing (a transposed V column: one
// byte per beat) first reads its beat (AXI ID 1, as port A) and is written whole, so the
// controller never does its own ECC read-modify-write (which holds the channel for the read's
// whole latency). The queue holds WQD beats per channel, so that the reads of a transposed V
// column's beats overlap (a slot lives from its push to its write response: about a read and a
// write latency). A beat's read waits until every older queued beat with the same address has
// its write response: a counter per bucket (a hash of the beat address) counts the live beats
// past the read point, and a read waits while its bucket's count is not zero (a collision only
// delays it). The queue's writes wait while its reads are still going out (unless it is half
// full), so the controller sees runs of reads, then runs of writes. A reads that fall in the beat of the previous A read (the MXU's scale stream:
// 16 scales per beat) reuse it without a DRAM access, until any write is accepted. An A read that
// misses fetches a run of up to APF channel-consecutive beats (one INCR burst, not across 4 KB);
// the A reads that follow the run in order take its beats without a DRAM access, and a read off
// the run (or after a B or A write) drops the beats not yet taken. An SW write drops the run
// (and the reused beat) only when it touches them: checked when the write is taken and again
// when its beat is in memory (a run fetched in between read the old beat), so the QSTs that
// stream while an MM runs do not cost its scale stream its runs. (A read that depends on a QST
// comes after the QST is done, i.e. after its writes' responses.) The scale stream thus costs
// one AXI transaction per APF beats instead of one per beat.
// Port B reads that follow each other in the address space (a streamed operand) are issued as
// one burst per channel: a run of queued contiguous reads goes out once it has BL beats, once
// the next queued request does not continue it (or it would cross a 4 KB page), once no request
// has arrived for GATHER cycles, or at once when no B read is in flight on the channel. A
// burst's beats reserve response room together, so the beats in flight stay within RD (each
// transaction has a fixed cost in the interconnect and the controller: single-beat reads
// reached about a quarter of the channel's bandwidth on the board).
// Requests are taken when req && rdy; rdy depends on registered state only. Reads return in
// order per port (the B tag with its data). Responses never back up: a read beat is issued on
// AXI only when its response FIFO has room reserved. wr_idle: every accepted write has its
// AXI write response.
module otpu_axi_dram #(
  parameter int D = 128,
  parameter int QD = 16,                             // request queue depth per channel and port
  parameter int WQD = 64,                            // SW (gathered beat) queue depth per channel
  parameter int BL = 8,                              // port B read burst, beats (max)
  parameter int GATHER = 4,                          // idle cycles before a short burst goes out
  parameter int WGATHER = 4,                         // idle cycles before a gathered SW beat goes out
  parameter int RD = 128,                            // B read beats in flight per channel
  parameter int AD = 32,                             // A read beats in flight per channel
  parameter int APF = 8,                             // A read run (prefetch), beats
  parameter logic [31:0] BASE0 = 32'h0000_0000,
  parameter logic [31:0] BASE1 = 32'h8000_0000
) (
  input  logic              clk,
  input  logic              rst,
  // slice side
  output logic              a_rdy,
  input  logic              a_req,
  input  logic              a_we,
  input  logic [31:0]       a_addr,     // word address
  input  logic [31:0]       a_wdata,
  input  logic [3:0]        a_be,
  output logic              a_rvalid,
  output logic [31:0]       a_rdata,
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
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  output logic              wr_idle,
  // AXI4 masters, one per channel (ID 0: port B, ID 1: port A)
  output logic [1:0]            m_awvalid,
  input  logic [1:0]            m_awready,
  output logic [1:0][31:0]      m_awaddr,
  output logic [1:0]            m_awid,
  output logic [1:0]            m_wvalid,
  input  logic [1:0]            m_wready,
  output logic [1:0][511:0]     m_wdata,
  output logic [1:0][63:0]      m_wstrb,
  input  logic [1:0]            m_bvalid,
  output logic [1:0]            m_bready,
  input  logic [1:0]            m_bid,
  input  logic [1:0][1:0]       m_bresp,
  output logic [1:0]            m_arvalid,
  input  logic [1:0]            m_arready,
  output logic [1:0][31:0]      m_araddr,
  output logic [1:0][7:0]       m_arlen,
  output logic [1:0]            m_arid,
  input  logic [1:0]            m_rvalid,
  output logic [1:0]            m_rready,
  input  logic [1:0]            m_rid,
  input  logic [1:0][511:0]     m_rdata,
  input  logic [1:0][1:0]       m_rresp,
  input  logic [1:0]            m_rlast,
  output logic                  err         // sticky: an AXI error response
);
  initial if (D != 128) $fatal(1, "otpu_axi_dram: D must be 128 (one beat per channel)");
  initial if (BL < 1 || BL > QD || BL > RD || BL > 16) $fatal(1, "otpu_axi_dram: bad BL");
  initial if (APF < 1 || APF > 8 || APF > AD) $fatal(1, "otpu_axi_dram: bad APF");
  localparam int LW = $clog2(BL + 1);
  localparam int QW = $clog2(QD);
  localparam int WW = $clog2(WQD);
  localparam int NH = 64;                            // SW hazard buckets (see w_blk)
  localparam int HW = $clog2(NH);
  localparam int RW = $clog2(RD);
  localparam int AW_ = $clog2(AD);
  localparam int OD = 2 * RD;                        // B tags / A order entries in flight (2^k)
  localparam int OW = $clog2(OD);

  // the hazard bucket of a beat (channel address bits 31:6): an XOR fold, so that the strided
  // beats of a transposed V column spread over the buckets
  function automatic logic [5:0] whash(input logic [25:0] a);
    return a[5:0] ^ a[11:6] ^ a[17:12] ^ a[23:18] ^ {4'd0, a[25:24]};
  endfunction

  function automatic logic [31:0] chan_addr(input logic [31:0] word_addr, input logic c);
    logic [31:0] beat;
    beat = word_addr >> 4;
    return (c ? BASE1 : BASE0) + ((beat >> 1) << 6);
  endfunction

  // ------------------------------------------------------------------ request queues
  // per channel: qb (port B beats), qa (port A beats)
  typedef struct packed {
    logic         we;
    logic [31:0]  addr;        // AXI address
    logic [511:0] data;
    logic [15:0]  wmask;       // word enables
  } qb_t;
  typedef struct packed {
    logic         we;
    logic [31:0]  addr;
    logic [3:0]   idx;         // word in the beat
    logic [31:0]  data;
    logic [3:0]   be;
    logic [3:0]   len;         // read: beats of the run
  } qa_t;
  typedef struct packed {      // a gathered SW beat
    logic [31:0]  addr;
    logic [511:0] data;
    logic [63:0]  strb;
  } qw_t;

  logic [QW:0] qb_n [2];
  logic [QW-1:0] qb_h [2];
  logic [QW:0] qa_n [2];
  logic [QW-1:0] qa_h [2];
  // port SW (gathered beats): live entries (until their write response) from qw_f; qw_h the
  // next to write, qw_r the next whose read is due (qw_wb written, qw_rn from qw_r on)
  logic [WW:0] qw_n [2], qw_wb [2], qw_rn [2];
  logic [WW-1:0] qw_f [2], qw_h [2], qw_r [2];
  logic [WQD-1:0] wpart [2], wgot [2];      // per slot: bytes missing; its read data is in
  // read-after-write hazards: whb counts, per bucket (a hash of the beat), the live entries past
  // the read point (from qw_f up to qw_r: read, or whole); wnz: whb != 0
  logic [WW:0]  whb [2][NH];
  logic [NH-1:0] wnz [2];
  // ID 1 in flight, in order per channel: reads (A 0 / SW-queue 1) and writes (A 0 / queue 1);
  // rix: the slots of the queue's reads in flight
  localparam int KD = 2 ** $clog2(AD + WQD);
  localparam int KW = $clog2(KD);
  logic [KD-1:0] k1r [2], k1w [2];
  logic [KW:0]  k1r_h [2], k1r_n [2], k1w_h [2], k1w_n [2];
  logic [WW-1:0] rix_o [2];                      // rix at rix_h: the oldest SW read's slot
  logic [WW-1:0] rix_h [2];
  logic [WW:0]  rix_n [2];
  // SW gather buffers (see the top): valid, beat, bytes written, cycles since the last merge
  logic [1:0]   gv;
  qw_t          gb [2];
  logic [2:0]   gage [2];

  // order of B reads (tags) and of A reads (channel, word, reuse)
  logic         bt_q [OD];                     // LUT RAM
  logic [OW:0]  bt_n;
  logic [OW-1:0] bt_h;
  // A reads in order: channel, word, reuse of the last beat, run beats to drop before this one
  typedef struct packed { logic c; logic [3:0] idx; logic reuse; logic [2:0] drop; } ao_t;
  ao_t          ao_q [OD];
  ao_t          aoh;                           // the oldest A read
  logic [OW:0]  ao_n;
  logic [OW-1:0] ao_h;

  // B read runs: qc marks a queued B read that continues the one queued before it (the next
  // chunk, same 4 KB page on the channels: a channel page is 64 chunks, b_addr[10:5]);
  // lb_rd / lb_nx: the last taken B request was a read, and the chunk that would continue it
  logic [QD-1:0] qc [2];
  logic         lb_rd;
  logic [31:0]  lb_nx;
  wire          b_cont = !b_we && lb_rd && b_addr == lb_nx && b_addr[10:5] != 0;
  logic [2:0]   qi [2];                         // cycles since the last B push (saturating)

  // A beat reuse
  logic         al_v;
  logic [27:0]  al_beat;
  wire  [27:0]  a_beat = a_addr[31:4];

  assign b_rdy = (qb_n[0] < QD) && (qb_n[1] < QD) && (bt_n < OD);
  assign a_rdy = (qa_n[0] < QD) && (qa_n[1] < QD) && (ao_n < OD);
  assign sw_rdy = (qw_n[0] < WQD) && (qw_n[1] < WQD);
  wire sw_take = sw_req && sw_rdy;
  wire sw_ch = sw_addr[4];
  wire b_take = b_req && b_rdy;
  wire a_take = a_req && a_rdy;
  wire a_ch = a_addr[4];
  wire a_reuse = !a_we && al_v && al_beat == a_beat;
  // A runs per channel: the next channel beat of the run, valid (no write since), beats not
  // yet taken (in flight or stored)
  logic [26:0]  pnx [2];
  logic [1:0]   pv;
  logic [2:0]   pfl [2];
  wire  [26:0]  a_cb = a_addr[31:5];              // the channel beat
  wire  a_hit = !a_we && !a_reuse && pv[a_ch] && a_cb == pnx[a_ch] && pfl[a_ch] != 0;
  // an SW write (taken, or in memory) that touches channel c's run or the reused beat drops them
  logic [1:0]   sw_kr, sw_ka;
  logic [25:0]  wadr_f [2];                       // the beat of slot qw_f (its write is done)
  always_comb begin
    for (int c = 0; c < 2; c++) begin
      logic [26:0] tb, lb;
      logic tk, ld;
      tb = sw_addr[31:5];                         // the taken write's channel beat
      lb = 27'(wadr_f[c] - (c ? BASE1[31:6] : BASE0[31:6]));
      tk = sw_take && sw_ch == c[0];
      ld = m_bvalid[c] && m_bid[c] && k1w[c][k1w_h[c][KW-1:0]];
      sw_kr[c] = pv[c] && ((tk && 27'(tb - pnx[c]) < 27'(pfl[c])) ||
                           (ld && 27'(lb - pnx[c]) < 27'(pfl[c])));
      sw_ka[c] = al_v && ((tk && al_beat == {tb, c[0]}) || (ld && al_beat == {lb, c[0]}));
    end
  end
  wire  [3:0]   a_len = (7'd64 - {1'b0, a_cb[5:0]} < 7'(APF)) ? 4'(7'd64 - {1'b0, a_cb[5:0]})
                                                               : 4'(APF);

  // ------------------------------------------------------------------ response FIFOs
  logic [RW:0]  rb_n [2], rb_res [2];          // stored; stored + in flight
  logic [RW-1:0] rb_h [2], rb_t [2];
  logic [AW_:0] ra_n [2], ra_res [2];
  logic [AW_-1:0] ra_h [2], ra_t [2];

  // ------------------------------------------------------------------ queue and FIFO memories
  // (per channel; written in their own processes so they map to LUT RAM)
  logic [1:0]   qb_push, qa_push, qw_push;
  logic [1:0]   ar_w;                            // the SW queue's read of a partial beat
  qb_t          qb_e [2];
  qa_t          qa_e [2];
  qw_t          qw_e [2];
  qb_t          hb [2];
  qa_t          ha [2];
  qw_t          hw [2];
  logic [25:0]  wadr_r [2];                        // the beat of slot qw_r
  logic [HW-1:0] wh_r [2], wh_f [2];               // the buckets of slots qw_r and qw_f
  logic [511:0] rb_head [2], ra_head [2], wr_head [2];
  always_comb begin
    for (int c = 0; c < 2; c++) begin
      qb_push[c] = b_take && (!b_we || b_wmask[16 * c +: 16] != 0);
      qb_e[c].we = b_we;
      qb_e[c].addr = chan_addr(b_addr + 32'(16 * c), c[0]);
      qb_e[c].data = b_wdata[512 * c +: 512];
      qb_e[c].wmask = b_we ? b_wmask[16 * c +: 16] : '0;
      qa_push[c] = a_take && a_ch == c[0] && !a_reuse && !a_hit;
      qa_e[c].we = a_we;
      qa_e[c].addr = chan_addr(a_addr, c[0]);
      qa_e[c].idx = a_addr[3:0];
      qa_e[c].data = a_wdata;
      qa_e[c].be = a_be;
      qa_e[c].len = a_we ? 4'd1 : a_len;
      // the gathered beat goes out: another beat's SW write (room is sure: sw_rdy), or, with
      // no SW write this cycle and room in the queue, full or idle
      qw_push[c] = gv[c] && (sw_take && sw_ch == c[0] ? gb[c].addr != chan_addr(sw_addr, c[0])
                   : qw_n[c] < WQD && (&gb[c].strb || gage[c] >= 3'(WGATHER)));
      qw_e[c] = gb[c];
    end
  end
  always_ff @(posedge clk)
    for (int c = 0; c < 2; c++) if (qb_push[c]) qc[c][QW'(qb_h[c] + qb_n[c])] <= b_cont;
  for (genvar c = 0; c < 2; c++) begin : g_mem
    // flat vectors: Vivado builds a RAM of structs from registers (qbm and qwm were ~37K
    // flip-flops and their read multiplexers, in the congested corner by the memory ports)
    logic [$bits(qb_t)-1:0] qbm [QD];
    logic [$bits(qa_t)-1:0] qam [QD];
    logic [$bits(qw_t)-1:0] qwm [WQD];
    logic [25:0] wam [WQD];                 // per slot: the beat (address bits 31:6)
    logic [HW-1:0] whm [WQD];               // per slot: its hazard bucket
    logic [511:0] rbm [RD];
    logic [511:0] ram [AD];
    always_ff @(posedge clk) if (qb_push[c]) qbm[QW'(qb_h[c] + qb_n[c])] <= qb_e[c];
    always_ff @(posedge clk) if (qa_push[c]) qam[QW'(qa_h[c] + qa_n[c])] <= qa_e[c];
    always_ff @(posedge clk) if (qw_push[c]) qwm[WW'(qw_f[c] + qw_n[c])] <= qw_e[c];
    always_ff @(posedge clk) if (qw_push[c]) wam[WW'(qw_f[c] + qw_n[c])] <= qw_e[c].addr[31:6];
    always_ff @(posedge clk) if (qw_push[c]) whm[WW'(qw_f[c] + qw_n[c])] <= whash(qw_e[c].addr[31:6]);
    assign wadr_r[c] = wam[qw_r[c]];
    assign wadr_f[c] = wam[qw_f[c]];
    assign wh_r[c] = whm[qw_r[c]];
    assign wh_f[c] = whm[qw_f[c]];
    always_ff @(posedge clk)
      if (m_rvalid[c] && !m_rid[c]) rbm[rb_t[c]] <= m_rdata[c];
    logic [WW-1:0] rixm [WQD];              // the slots of the queue's reads in flight
    always_ff @(posedge clk)
      if (m_arvalid[c] && m_arready[c] && ar_w[c]) rixm[WW'(rix_h[c] + rix_n[c])] <= qw_r[c];
    assign rix_o[c] = rixm[rix_h[c]];
    logic [511:0] wrm [WQD];                // the queue's read data, per slot
    always_ff @(posedge clk)
      if (m_rvalid[c] && m_rid[c] && !k1r[c][k1r_h[c][KW-1:0]]) ram[ra_t[c]] <= m_rdata[c];
    always_ff @(posedge clk)
      if (m_rvalid[c] && m_rid[c] && k1r[c][k1r_h[c][KW-1:0]]) wrm[rix_o[c]] <= m_rdata[c];
    assign wr_head[c] = wrm[qw_h[c]];
    assign hb[c] = qb_t'(qbm[qb_h[c]]);
    assign ha[c] = qa_t'(qam[qa_h[c]]);
    assign hw[c] = qw_t'(qwm[qw_h[c]]);
    assign rb_head[c] = rbm[rb_h[c]];
    assign ra_head[c] = ram[AW_'(ra_h[c] + AW_'(aoh.drop))];
  end

  // ------------------------------------------------------------------ per-channel issue
  logic [1:0] ar_b, ar_a, w_b, w_a, w_w;         // this cycle's AR / write source
  logic [LW-1:0] run [2];                        // B reads queued at the head, contiguous
  logic [1:0] arh, arh_a, arh_w;                 // an AR shown and not taken: held (A, SW, B)
  logic [1:0] w_blk, w_hold;                     // its read waits (same beat older); writes wait
  logic [LW-1:0] arh_n [2];                      // the held B burst's beats
  logic [1:0] aw_done, w_done;                   // current write head: halves already taken
  logic [1:0] wsrc [2];                          // current write's source: 0 qb, 1 qa, 2 qw
  logic [1:0] wcur;                              // a write is in progress
  qa_t hs [2];
  always_comb begin
    for (int c = 0; c < 2; c++) begin
      // reads: A first (rare), then B; each needs reserved response room. A B read waits for
      // its run to fill (see the top); an AR shown on the bus stays as it is until taken
      begin
        logic stop, go;
        run[c] = LW'(1);
        stop = 1'b0;
        for (int k = 1; k < BL; k++)
          if (!stop && (QW + 1)'(k) < qb_n[c] && qc[c][QW'(qb_h[c] + QW'(k))]) run[c] = LW'(k + 1);
          else stop = 1'b1;
        go = run[c] == LW'(BL) || (QW + 1)'(run[c]) < qb_n[c] || qi[c] >= 3'(GATHER) ||
             rb_res[c] == 0;
        // the SW queue's next partial beat: blocked while an older live entry may have its
        // address (one in its bucket; a hash collision only delays the read)
        w_blk[c] = wnz[c][wh_r[c]];
        if (arh[c]) begin
          ar_a[c] = arh_a[c];
          ar_w[c] = arh_w[c];
          ar_b[c] = !arh_a[c] && !arh_w[c];
          run[c] = arh_n[c];
        end else begin
          ar_a[c] = (qa_n[c] != 0) && !ha[c].we && (ra_res[c] <= (AW_ + 1)'(AD - ha[c].len)) &&
                    (k1r_n[c] < (KW + 1)'(KD));
          ar_w[c] = !ar_a[c] && (qw_rn[c] != 0) && wpart[c][qw_r[c]] && !w_blk[c] &&
                    (k1r_n[c] < (KW + 1)'(KD));
          ar_b[c] = !ar_a[c] && !ar_w[c] && (qb_n[c] != 0) && !hb[c].we && go &&
                    rb_res[c] <= (RW + 1)'(RD - run[c]);
        end
      end
      m_arvalid[c] = ar_a[c] || ar_w[c] || ar_b[c];
      m_araddr[c] = ar_a[c] ? ha[c].addr : ar_w[c] ? {wadr_r[c], 6'd0} : hb[c].addr;
      m_arlen[c] = ar_a[c] ? 8'(ha[c].len - 1) : ar_w[c] ? 8'd0 : 8'(run[c] - 1);
      m_arid[c] = ar_a[c] || ar_w[c];
      // the queue's writes wait while its reads still go out (not blocked), unless half full
      w_hold[c] = (qw_rn[c] != 0) && wpart[c][qw_r[c]] && !w_blk[c] && qw_n[c] < (WW + 1)'(WQD / 2);
      // writes: the head write of qw, qa or qb (kept until both AW and W are taken)
      if (wcur[c]) begin
        w_w[c] = (wsrc[c] == 2'd2);
        w_a[c] = (wsrc[c] == 2'd1);
        w_b[c] = (wsrc[c] == 2'd0);
      end else begin
        w_w[c] = (qw_n[c] != qw_wb[c]) && (!wpart[c][qw_h[c]] || wgot[c][qw_h[c]]) &&
                 !w_hold[c] && (k1w_n[c] < (KW + 1)'(KD));
        w_a[c] = !w_w[c] && (qa_n[c] != 0) && ha[c].we && (k1w_n[c] < (KW + 1)'(KD));
        w_b[c] = !w_w[c] && !w_a[c] && (qb_n[c] != 0) && hb[c].we;
      end
      hs[c] = ha[c];
      m_awvalid[c] = (w_w[c] || w_a[c] || w_b[c]) && !aw_done[c];
      m_wvalid[c] = (w_w[c] || w_a[c] || w_b[c]) && !w_done[c];
      m_awaddr[c] = w_w[c] ? hw[c].addr : w_a[c] ? hs[c].addr : hb[c].addr;
      m_awid[c] = w_w[c] || w_a[c];
      m_wdata[c] = '0;
      m_wstrb[c] = '0;
      if (w_w[c]) begin
        for (int k = 0; k < 64; k++)
          m_wdata[c][8 * k +: 8] = hw[c].strb[k] ? hw[c].data[8 * k +: 8] : wr_head[c][8 * k +: 8];
        m_wstrb[c] = '1;
      end else if (w_a[c]) begin
        m_wdata[c][32 * hs[c].idx +: 32] = hs[c].data;
        m_wstrb[c][4 * hs[c].idx +: 4] = hs[c].be;
      end else begin
        m_wdata[c] = hb[c].data;
        for (int k = 0; k < 16; k++) m_wstrb[c][4 * k +: 4] = {4{hb[c].wmask[k]}};
      end
      m_bready[c] = 1'b1;
      m_rready[c] = 1'b1;
    end
  end

  // ------------------------------------------------------------------ merge
  wire b_out = (bt_n != 0) && (rb_n[0] != 0) && (rb_n[1] != 0);
  assign aoh = ao_q[ao_h];
  logic [511:0] a_last;                          // the beat of the last fetched A read
  wire a_out = (ao_n != 0) && (aoh.reuse || ra_n[aoh.c] > (AW_ + 1)'(aoh.drop));
  wire [511:0] a_src = aoh.reuse ? a_last : ra_head[aoh.c];
  assign b_rvalid = b_out;
  assign b_rtag = bt_q[bt_h];
  assign b_rdata = {rb_head[1], rb_head[0]};
  assign a_rvalid = a_out;
  assign a_rdata = a_src[32 * aoh.idx +: 32];

  always_ff @(posedge clk) begin
    if (b_take && !b_we) bt_q[OW'(bt_h + bt_n)] <= b_tag;
    if (a_take && !a_we)
      ao_q[OW'(ao_h + ao_n)] <= '{c: a_ch, idx: a_addr[3:0], reuse: a_reuse,
                                  drop: (a_reuse || a_hit) ? 3'd0 : pfl[a_ch]};
  end

  // ------------------------------------------------------------------ writes outstanding
  // wr_n counts the writes accepted before the last cycle; wacc_q holds the last cycle's accepts
  // (folded into wr_n a cycle late, off the rdy -> take path). A B response comes after its W
  // handshake, so at least a cycle after the accept: wr_n never goes negative
  logic [15:0] wr_n;
  logic [4:0]  wacc_q;
  assign wr_idle = (wr_n == 0) && (wacc_q == '0) && (gv == '0);

  always_ff @(posedge clk) begin
    if (rst) begin
      for (int c = 0; c < 2; c++) begin
        qb_n[c] <= '0; qb_h[c] <= '0; qa_n[c] <= '0; qa_h[c] <= '0;
        qw_n[c] <= '0; qw_h[c] <= '0; qw_f[c] <= '0; qw_r[c] <= '0; qw_wb[c] <= '0;
        qw_rn[c] <= '0; rix_h[c] <= '0; rix_n[c] <= '0; wnz[c] <= '0;
        for (int k = 0; k < NH; k++) whb[c][k] <= '0;
        k1r_h[c] <= '0; k1r_n[c] <= '0; k1w_h[c] <= '0; k1w_n[c] <= '0;
        rb_n[c] <= '0; rb_res[c] <= '0; rb_h[c] <= '0; rb_t[c] <= '0;
        ra_n[c] <= '0; ra_res[c] <= '0; ra_h[c] <= '0; ra_t[c] <= '0;
      end
      bt_n <= '0; bt_h <= '0; ao_n <= '0; ao_h <= '0;
      al_v <= 1'b0;
      pv <= '0; pfl[0] <= '0; pfl[1] <= '0;
      wr_n <= '0; wacc_q <= '0;
      gv <= '0; gage[0] <= '0; gage[1] <= '0;
      aw_done <= '0; w_done <= '0; wcur <= '0;
      arh <= '0; arh_w <= '0; lb_rd <= 1'b0;
      qi[0] <= '0; qi[1] <= '0;
      wsrc[0] <= '0; wsrc[1] <= '0;
      err <= 1'b0;
    end else begin
      // writes outstanding: the last cycle's (registered) accepts and this cycle's responses fold
      // into one small delta (a sum of single bits), added once. Accepts are counted from the
      // take strobes, not the queue pushes: an A write is never a reuse and goes to exactly one
      // channel, as does an SW write, so a_reuse and the channel decode stay off this path
      logic signed [4:0] wn;
      wn = 5'(wacc_q[0]) + 5'(wacc_q[1]) + 5'(wacc_q[2]) + 5'(wacc_q[3]) + 5'(wacc_q[4]) -
           5'(m_bvalid[0]) - 5'(m_bvalid[1]);
      wacc_q <= {qw_push[1], qw_push[0], a_take && a_we,
                 b_take && b_we && b_wmask[31:16] != 0,
                 b_take && b_we && b_wmask[15:0]  != 0};
      for (int c = 0; c < 2; c++) begin
        logic [QW:0] nb, na;
        logic [WW:0] nw;
        logic [RW:0] rbn, rbr;
        logic [AW_:0] ran, rar;
        logic popb, popa, popw;
        logic [LW-1:0] popn;
        logic [WW:0] nwb, nrn, nrx;
        logic [KW:0] nkr, nkw;
        logic hinc, hdec;
        nb = qb_n[c]; na = qa_n[c]; nw = qw_n[c];
        nwb = qw_wb[c]; nrn = qw_rn[c]; nrx = rix_n[c]; nkr = k1r_n[c]; nkw = k1w_n[c];
        rbn = rb_n[c]; rbr = rb_res[c]; ran = ra_n[c]; rar = ra_res[c];
        popb = 1'b0; popa = 1'b0; popw = 1'b0; popn = LW'(1);
        hinc = 1'b0; hdec = 1'b0;
        // ---- accept
        if (qb_push[c]) begin
          nb = nb + 1;
        end
        if (qa_push[c]) begin
          na = na + 1;
        end
        if (qw_push[c]) begin
          logic [WW-1:0] t;
          t = WW'(qw_f[c] + qw_n[c]);
          wpart[c][t] <= !(&qw_e[c].strb);
          wgot[c][t] <= 1'b0;
          nw = nw + 1;
          nrn = nrn + 1;
        end
        // ---- AR
        if (m_arvalid[c] && m_arready[c]) begin
          if (ar_a[c] || ar_w[c]) begin
            k1r[c][KW'(k1r_h[c] + k1r_n[c])] <= ar_w[c];
            nkr = nkr + 1;
          end
          if (ar_a[c]) begin popa = 1'b1; rar = rar + (AW_ + 1)'(ha[c].len); end
          else if (ar_w[c]) begin
            nrx = nrx + 1;
            qw_r[c] <= qw_r[c] + 1;
            nrn = nrn - 1;
          end
          else begin popb = 1'b1; popn = run[c]; rbr = rbr + (RW + 1)'(run[c]); end
        end
        // a whole beat needs no read (ar_w is never shown for one): it passes the read point at
        // once, so before its write (and response) can come
        if (qw_rn[c] != 0 && !wpart[c][qw_r[c]]) begin
          qw_r[c] <= qw_r[c] + 1;
          nrn = nrn - 1;
        end
        // (from ar_w, not m_arvalid: off the B run path)
        hinc = (ar_w[c] && m_arready[c]) || (qw_rn[c] != 0 && !wpart[c][qw_r[c]]);
        arh[c] <= m_arvalid[c] && !m_arready[c];
        arh_a[c] <= ar_a[c];
        arh_w[c] <= ar_w[c];
        arh_n[c] <= run[c];
        if (qb_push[c]) qi[c] <= '0;
        else if (qi[c] != '1) qi[c] <= qi[c] + 1;
        // ---- AW / W (a write leaves its queue once both are taken)
        if (w_w[c] || w_a[c] || w_b[c]) begin
          logic awd, wd;
          awd = aw_done[c] || (m_awvalid[c] && m_awready[c]);
          wd = w_done[c] || (m_wvalid[c] && m_wready[c]);
          if (awd && wd) begin
            if (w_w[c]) popw = 1'b1; else if (w_a[c]) popa = 1'b1; else popb = 1'b1;
            if (w_w[c] || w_a[c]) begin
              k1w[c][KW'(k1w_h[c] + k1w_n[c])] <= w_w[c];
              nkw = nkw + 1;
            end
            aw_done[c] <= 1'b0; w_done[c] <= 1'b0; wcur[c] <= 1'b0;
          end else begin
            aw_done[c] <= awd; w_done[c] <= wd; wcur[c] <= 1'b1;
            wsrc[c] <= w_w[c] ? 2'd2 : w_a[c] ? 2'd1 : 2'd0;
          end
        end
        if (popb) begin qb_h[c] <= qb_h[c] + QW'(popn); nb = nb - (QW + 1)'(popn); end
        if (popa) begin qa_h[c] <= qa_h[c] + 1; na = na - 1; end
        if (popw) begin qw_h[c] <= qw_h[c] + 1; nwb = nwb + 1; end
        // ---- R
        if (m_rvalid[c]) begin
          if (m_rresp[c][1]) err <= 1'b1;
          if (m_rid[c]) begin
            if (k1r[c][k1r_h[c][KW-1:0]]) begin  // the SW queue's read: its slot has the data
              wgot[c][rix_o[c]] <= 1'b1;
              rix_h[c] <= rix_h[c] + 1;
              nrx = nrx - 1;
            end else begin
              ra_t[c] <= ra_t[c] + 1;
              ran = ran + 1;
            end
            if (m_rlast[c]) begin                // (an A run is several beats)
              k1r_h[c] <= k1r_h[c] + 1;
              nkr = nkr - 1;
            end
          end else begin
            rb_t[c] <= rb_t[c] + 1;
            rbn = rbn + 1;
          end
        end
        // ---- B
        if (m_bvalid[c]) begin
          if (m_bresp[c][1]) err <= 1'b1;
          if (m_bid[c]) begin
            if (k1w[c][k1w_h[c][KW-1:0]]) begin  // the SW queue's oldest write is done
              qw_f[c] <= qw_f[c] + 1;
              hdec = 1'b1;
              nw = nw - 1;
              nwb = nwb - 1;
            end
            k1w_h[c] <= k1w_h[c] + 1;
            nkw = nkw - 1;
          end
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
        // ---- hazard buckets: slot qw_r passes the read point, slot qw_f's write is done
        if (hinc && !(hdec && wh_f[c] == wh_r[c])) begin
          whb[c][wh_r[c]] <= whb[c][wh_r[c]] + 1'b1;
          wnz[c][wh_r[c]] <= 1'b1;
        end
        if (hdec && !(hinc && wh_f[c] == wh_r[c])) begin
          whb[c][wh_f[c]] <= whb[c][wh_f[c]] - 1'b1;
          wnz[c][wh_f[c]] <= whb[c][wh_f[c]] != (WW + 1)'(1);
        end
        qb_n[c] <= nb; qa_n[c] <= na; qw_n[c] <= nw;
        qw_wb[c] <= nwb; qw_rn[c] <= nrn; rix_n[c] <= nrx; k1r_n[c] <= nkr; k1w_n[c] <= nkw;
        rb_n[c] <= rbn; rb_res[c] <= rbr; ra_n[c] <= ran; ra_res[c] <= rar;
      end
      wr_n <= wr_n + {{11{wn[4]}}, wn};
      // ---- order FIFOs
      begin
        logic [OW:0] btn, aon;
        btn = bt_n; aon = ao_n;
        if (b_take && !b_we) btn = btn + 1;
        if (b_out) begin bt_h <= bt_h + 1; btn = btn - 1; end
        if (a_take && !a_we) aon = aon + 1;
        if (a_out) begin
          ao_h <= ao_h + 1; aon = aon - 1;
          if (!aoh.reuse) a_last <= ra_head[aoh.c];
        end
        bt_n <= btn; ao_n <= aon;
      end
      // ---- B runs: the last taken B request
      if (b_take) begin
        lb_rd <= !b_we;
        lb_nx <= b_addr + 32'(D / 4);
      end
      // ---- A beat reuse: the beat of the last A read, forgotten on any write
      // ---- SW gather: merge into the beat, or start a new one (the old one was pushed)
      for (int c = 0; c < 2; c++) begin
        if (sw_take && sw_ch == c[0]) begin
          logic same;
          same = gv[c] && gb[c].addr == chan_addr(sw_addr, c[0]);
          gv[c] <= 1'b1;
          gage[c] <= '0;
          gb[c].addr <= chan_addr(sw_addr, c[0]);
          for (int k = 0; k < 64; k++) begin
            logic hit;
            hit = sw_addr[3:0] == 4'(k / 4) && sw_be[k % 4];
            gb[c].data[8 * k +: 8] <= hit ? sw_wdata[8 * (k % 4) +: 8]
                                     : same ? gb[c].data[8 * k +: 8] : 8'h00;
            gb[c].strb[k] <= hit || (same && gb[c].strb[k]);
          end
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
          pfl[a_ch] <= 3'(a_len - 1);
          pv[a_ch] <= 1'b1;
        end
      end
      for (int c = 0; c < 2; c++) if (sw_kr[c]) pv[c] <= 1'b0;
      if ((b_take && b_we) || (a_take && a_we)) pv <= '0;
      if ((b_take && b_we) || (a_take && a_we) || (|sw_ka)) al_v <= 1'b0;
      else if (a_take && !a_we) begin
        al_v <= 1'b1;
        al_beat <= a_beat;
      end
    end
  end
endmodule
