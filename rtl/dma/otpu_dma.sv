// LD / ST: move 32-bit words between the slice DRAM and TMEM using the DRAM burst port (B).
// DRAM side: every D-byte chunk the transfer touches is requested once (B requests are chunk
// aligned). TMEM side: the transfer is cut into W = min(D/4, LANES)-word segments aligned in
// DRAM, one per cycle (CW/W per chunk), so the lanes need no shifter: lane l carries word l of
// the segment, and the partial segments at the ends are masked. The segment's words sit in
// consecutive TMEM words, hence distinct banks.
//   LD  the chunks are requested ahead into a DEPTH-chunk buffer (block RAM). A chunk's slot is
//       reserved when it is requested, so read data (which cannot be refused) always has room.
//       A segment leaves the buffer through its registered read and the TMEM write register.
//   ST  the segments read from TMEM are gathered into a chunk register; a chunk is complete
//       with its last segment in range (that segment comes straight from TMEM) and goes, with
//       its word mask, into the LD chunk buffer (idle during an ST). The buffered chunks are
//       written in runs of SRUN (the rest at the end) on consecutive cycles: the DMA has
//       priority on port B, so no MXU read comes between them, and the DRAM sees one read /
//       write turnaround per run instead of one per chunk (a chunk every SPC cycles, the MXU's
//       weight reads in between).
//   DSTEP / STREAM  a stream (docs/stream.md): an fp32 matrix in DRAM passed row by row through
//       the stream engine (SE, the VPU's slot-0 reduction and its tail otpu_se_tail, in
//       otpu_vpu) and written back (DSTEP: one Gated DeltaNet head step, docs/isa.md). The
//       DMA holds SE for the whole instruction (ss_req, granted with ss_gnt once SE has no
//       VOP in flight). A STREAM first reads its descriptor (8 TMEM words) into ss_cfg and the
//       addresses; DSTEP's are its fields (GDN). The stream's vectors and scalars are then read
//       from TMEM into SE (the fill, ss_fk); then the stream's chunks are requested like an
//       LD's and their segments fed to SE, one per cycle, and the updated segments gathered
//       into chunks and written back; finally the row outputs o go to TMEM through the write
//       register. The DRAM sees runs: the stream's reads go out RUN chunks at a time (one
//       burst per channel) and the writes once RUN chunks are gathered (NGC-chunk gather), so
//       a read / write turnaround and a read transaction's fixed cost are paid per run, not
//       per chunk; a run of one kind never starts inside a run of the other. SE advances with
//       `pe` (ss_pe), a register: a cycle it may take a segment (or, after the last one, a
//       bubble) needs a segment in the buffer and room for two more updated segments.
//       With SR > 1 (an engine of its own with SR row groups, otpu_se_eng; docs/se_engine.md)
//       a pe cycle takes the same segment of SR consecutive rows (a row step: rows SR i .. SR i +
//       SR - 1; the last step's missing rows are +0, their outputs dropped), and the chunks go in
//       and out in that order: per row step, per chunk column j, per row (the reads' and the
//       writes' addresses step so). The buffer stays a FIFO of chunks; its chunks are taken into
//       one register per row (the next set) as they come, and a set moves to the current one
//       (the segments fed) as the one before is used up; the updated segments collect into one
//       chunk register per row and go into the gather a chunk a cycle; the row outputs come SR a
//       pe cycle (consecutive rows: SR lanes of the o buffer).
//       With ST16 a stream's flag DF_ST16 (DSTEP) / STF_ST16 (STREAM) keeps its DRAM matrix in
//       16 bits (docs/isa.md "ST16"): ST16 1 e6m9, 2 bf16. A chunk holds 2 SPC segments of W
//       16-bit values, each expanded to fp32 as it is fed (st16_x), and the updated segments
//       are rounded (to nearest even, st16_pack) into the chunks written: half the chunks each
//       way, the engine's arithmetic unchanged. The updated segments go out in pairs: the even
//       one's values held (yh), then a pair a word (SR = 1: into the gather; SR > 1: into the
//       row's chunk register at the fp32 positions, so its write data is one 2:1 select ahead
//       of the registers). A STREAM's flag STF_ST16R (STF_ST16W) keeps its source (its
//       destination) alone in 16 bits, the other end fp32 (dst apart from src): the reads
//       follow ds_hr, the writes ds_hw, each side's chunks a row its own (SR > 1: cpr_r / cpr_w).
//   WAITW  an LD of one word whose delivery waits for its condition (docs/isa.md): the word is
//       taken from the chunk as the LD would write it (C1), compared (C2), and written to TMEM
//       through lane 0 if it holds (C3); else the DMA waits w5 cycles and reads the chunk again.
//       Past w6 cycles (w6 != 0; ww_late) it stops with `err` (the slice's error, seen by the
//       host) between two reads: a read in flight at the timeout still ends the WAITW if its
//       word holds, so the WAITW either completes (word written) or times out (nothing written).
//   LDA  (LDA = 1, Config.QONCE; docs/isa.md "Quantize once") rows = w2[7:0] of a QACT ST's
//       bytes and records at R[ra] + w1 + r w3 (multiples of 64 bytes) into ACT rows w4 + r at
//       blocks ab = w2[15:8] .. ab + KB - 1 (KB = w2[31:16]): each row's KB x D bytes, then its
//       KB records of 8 bytes. A row is an LD's range whose segments go to ACT RAM instead of
//       TMEM: ALANES bytes a cycle (4 W / ALANES cycles a segment) and a record a cycle (W / 2),
//       through a queue (AQD entries) into otpu_actram's DMA port, which takes an entry in a cycle
//       with no quantizer write at its write stage (act_busy: the quantizer first; the delivery
//       waits while the queue is full). The requests go on to the next row as a row's last chunk
//       is requested (lq_*), the delivery as its last segment is delivered (la_*); the LDA stays
//       busy until the queue has drained, and is done two cycles after its last entry left.
// The DRAM port may refuse a request (b_gnt low) and read data may take any time to return (in
// order). An ST or a DSTEP completes once the DRAM has acknowledged all its writes (wr_idle).
// LDQ 2 (docs/dma_ldq.md): a second command in flight. A command started
// while the DMA is busy waits in a slot (pcmd), and rdy falls while the slot is taken (or an ST's
// done is pending: sb, below). A queued LD
// (of words) or LDA launches its chunk requests (the front: ic, cleft; an LDA's rows lq_*) under
// the command before it (the back: delivery, completion) once that is an LD or an LDA whose
// requests have all gone out, an LD also behind an ST in its wr_idle wait whose DRAM words it does
// not read; a queued ST (of words) starts behind an ST in its wr_idle wait (below); any other
// queued command starts once the DMA is idle. When the back completes, the
// front's command takes the back the same cycle (its delivery starts after the completed command's
// last TMEM write, an LDA's last ACT write: dones stay in issue order). The
// chunk buffer is shared: the front's chunks follow the back's (in order, each command eats the
// chunks it requested), so the returns need no tag; occ, counting the requests, keeps the front's
// within DEPTH. A WAITW, a DSTEP or a STREAM never runs beside another command.
// An ST behind an ST (go_s): the ST in its wait shrinks to sb, its done pending (done once
// wr_idle, as before; rdy stays low meanwhile: two commands in flight), and the queued ST takes
// the ST's machine at once: its TMEM reads fill the chunk buffer (empty in the wait), and its
// writes wait for that done (a write run starts once sb is clear or clears this cycle). wr_idle is
// the only acknowledgement there is: port B's writes are acknowledged in order per memory channel,
// not across them (otpu_native_dram's per-channel n_wdone; the QST's SW writes share them), so the
// back's done cannot count its own; held writes leave it the back's, and an ST's TMEM reads (CW /
// W cycles a chunk) are the slow side, so the buffer's chunks go out at a chunk a cycle after it.
module otpu_dma
  import otpu_pkg::*;
#(
  parameter int D     = 32,
  parameter int LANES = 8,
  parameter int DEPTH = 128,                          // LD chunk buffer (a power of two)
  parameter int LDQ   = 1,                            // commands in flight: 2, an LD's DRAM reads
                                                      //   under the command before it (below)
  parameter bit HAS_DSTEP = 1'b1,                     // streams: DSTEP, STREAM (SE's tail in u_vpu)
  parameter int SR    = 1,                            // stream rows a pe cycle (otpu_se_eng R)
  parameter bit DSTEP2 = 1'b0,                        // DSTEP's pair (flag PAIR, docs/isa.md)
  parameter int ST16  = 0,                            // a stream's DRAM matrix in 16 bits (flag
                                                      //   ST16): 1 e6m9, 2 bf16; 0 none
  parameter bit ST16S = 1'b0,                         // with ST16: STREAM's flags ST16R / ST16W
                                                      //   (one side alone 16-bit; Config.ST16S)
  parameter int AW    = 32,                           // DRAM byte address bits (otpu_seq AW)
  parameter bit LDA   = 1'b0,                         // LDA: DRAM -> ACT RAM (above)
  parameter int ALANES = 16,                          // ... its ACT write's bytes (otpu_actram LANES)
  parameter int WA    = waw(AW)                       // port B's word address bits (derived)
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    start,
  input  cmd_t                    cmd,
  output logic                    rdy,
  output logic                    done,
  output logic                    err,        // a WAITW timed out
  output logic                    ww_held,    // a WAITW's word held: one cycle, as it ends (below)
  // DRAM port B (the DMA has priority on it; read responses are routed back by tag)
  output logic                    b_req,
  input  logic                    b_gnt,      // the request is taken this cycle
  output logic                    b_we,
  output logic [D/4-1:0]          b_wmask,
  output logic [D*8-1:0]          b_wdata,
  output logic [WA-1:0]           b_addr,
  input  logic                    b_rvalid,
  input  logic [D*8-1:0]          b_rdata,
  input  logic                    wr_idle,    // no DRAM write outstanding
  // TMEM lanes (read port A, write port)
  output logic [LANES-1:0]        t_ren,
  output logic [LANES-1:0][31:0]  t_raddr,
  // t_ren by TMEM bank (otpu_tmem PBR: the port's rotator merge, from flip-flops as t_ren)
  output logic [LANES-1:0]        t_rben,
  output logic [LANES-1:0][31:0]  t_rbad,
  input  logic [LANES-1:0][31:0]  t_rdata,
  output logic [LANES-1:0]        t_wen,
  output logic [LANES-1:0][31:0]  t_waddr,
  output logic [LANES-1:0][31:0]  t_wdata,
  // the stream engine (otpu_vpu, docs/stream.md 8), held from ss_req until it falls; the fill
  // (ss_fk: SF_*, one step of W words) and the stream (ss_pe, ss_in_*) start after ss_gnt. SE
  // registers these a cycle; the updated segments (ss_y_*) and the row outputs (ss_o_*) come
  // back qualified with SE's pe (one each cycle ss_y_v / ss_o_v is up)
  output logic                    ss_req,
  input  logic                    ss_gnt,
  output ss_cfg_t                 ss_cfg,
  output logic                    ss_pe,
  output logic                    ss_in_v,
  output logic [31:0]             ss_in_d [SR*LANES],
  output logic [2:0]              ss_fk,
  output logic [4:0]              ss_fi,
  output logic [31:0]             ss_fd [LANES],
  input  logic                    ss_y_v,
  input  logic [31:0]             ss_y_d [SR*LANES],
  input  logic                    ss_o_v,
  input  logic [SR-1:0][31:0]     ss_o_d,
  // LDA: the quantizer's ACT RAM write at its write stage (it goes first), and the DMA's write
  // into otpu_actram's DMA port (d_*)
  input  logic                    act_busy,
  output logic [ALANES-1:0]       act_we,
  output logic [7:0]              act_row,
  output logic [31:0]             act_idx,
  output logic [ALANES-1:0][7:0]  act_data,
  output logic                    asc_we,
  output logic [7:0]              asc_row,
  output logic [15:0]             asc_blk,
  output logic [63:0]             asc_data
);
  localparam int CW  = D / 4;                         // words per chunk
  localparam int W   = (CW < LANES) ? CW : LANES;     // words per segment
  localparam int SPC = CW / W;                        // segments per chunk
  localparam int SWL = $clog2(W);
  localparam int CWL = $clog2(CW);
  localparam int PW  = $clog2(DEPTH);
  initial if (DEPTH != (1 << PW) || DEPTH < 2) $fatal(1, "otpu_dma: DEPTH must be a power of two");
  initial if (LDQ != 1 && LDQ != 2) $fatal(1, "otpu_dma: LDQ %0d (1 or 2)", LDQ);
  initial if (LDA && (ALANES < 8 || (ALANES & (ALANES - 1)) != 0 || (4 * W) % ALANES != 0 ||
                      4 * W > 64 || W < 2))
    $fatal(1, "otpu_dma: LDA needs ALANES a power of two >= 8 dividing 4 W, and 2 <= W <= 16");

  logic        busy, is_st, ackw;
  // LDQ 2 (the header): a start on a busy DMA (acc) puts the command in pcmd, queued (sl_q) until
  // the DMA is idle (go_q) or, an LD of words or an LDA (q_ld), until its requests may start under
  // the back (go_f: the slot is then the front, sl_f) unless they read words of an ST in the back
  // (q_ov; an LDA never goes behind an ST);
  // the back's completion hands the back over (hand). A queued ST (q_st) starts behind an ST in
  // its wr_idle wait (go_s), which leaves sb: its done pending. A start just after an LD's or a
  // WAITW's completion (ld_fin, an LDA's la_fin) is queued too: two dones never fall on one cycle.
  // rdy from flip-flops
  cmd_t        pcmd;
  logic        sl_q, sl_f, q_ld, q_ov, q_st, sb;
  logic        ld_fin;                 // LD: the last write is in the write register
  logic        la_fin;                 // LDA: its ACT queue drained
  logic        go_f, hand, go_s;
  assign rdy = (LDQ > 1) ? !(sl_q || sl_f || sb) : !busy;
  wire   acc  = LDQ > 1 && start && (busy || ld_fin || la_fin);
  wire   go_q = LDQ > 1 && sl_q && !busy && !ld_fin && !la_fin;
  wire   go   = (start && !acc) || go_q || go_s;   // a command starts the machine (go_s: busy)
  cmd_t  gc;                               // the command a go starts (go_f launches, hand hands over)
  assign gc = (LDQ > 1 && (sl_q || sl_f)) ? pcmd : cmd;
  // an LD's, an ST's or a WAITW's words (one carry chain each: the counts from the start's offset
  // in its segment / chunk): its DRAM words [a, e), the TMEM side (sw, so, sm, ns segments) and
  // the requests (ic, nc chunks)
  typedef struct packed {
    logic [WA-1:0] a, e, sw, ic;
    logic [31:0]   n, so, ns, nc;
    logic [W-1:0]  sm;
  } dec_t;
  function automatic dec_t cmd_dec(input cmd_t c);
    dec_t d;
    logic [31:0] ow, oc;
    d.a = WA'(AW'(cmd_a(c, 1)) >> 2);
    d.n = (c.op == OP_WAITW) ? 32'd1 : c.w3;     // WAITW: one word (w3 is its reference)
    if (LDA && c.op == OP_LDA)                     // LDA: a row's bytes and records
      d.n = (c.w2[7:0] == 0) ? '0 : ((32'(c.w2[31:16]) * D) >> 2) + (32'(c.w2[31:16]) << 1);
    ow = d.a % W;
    oc = d.a % CW;
    d.e = d.a + d.n;
    d.sw = d.a & ~WA'(W - 1);
    d.so = c.w2 - ow;
    for (int l = 0; l < W; l++) d.sm[l] = (32'(l) >= ow) && (d.n > 32'(l) - ow);
    d.ns = (d.n + (ow + (W - 1))) >> SWL;
    d.ic = d.a & ~WA'(CW - 1);
    d.nc = (d.n + (oc + (CW - 1))) >> CWL;
    return d;
  endfunction
  dec_t gd;
  assign gd = cmd_dec(gc);
  // DSTEP: the phases (fill: TMEM reads of the vectors; run: stream; out: o to TMEM)
  logic        is_ds, ds_fill, ds_run, ds_out, ds_zero;
  logic [WA-1:0] dw, de;               // the DRAM word range [dw, de)
  // TMEM side: the segment at DRAM word address sw, its TMEM address, its lanes in range
  // (registered: the TMEM arbiter sees these lanes), and the segments left
  logic [WA-1:0] sw;
  logic [31:0]  so, sleft;
  logic [W-1:0] sm;
  // sleft != 0 and sleft == 1, registered with sleft (as cl_nz / cl_one): an ST's TMEM read runs
  // from them into the TMEM arbiter and, through QUANT's grant, into otpu_native_dram's SW port
  logic         sl_nz, sl_one;

  function automatic logic in_rng(input logic [WA-1:0] a);
    return (a >= dw) && (a < de);
  endfunction
  function automatic int pos_of(input logic [WA-1:0] a);   // segment index within its chunk
    return int'((a % CW) / W);
  endfunction
  wire seg_end = (pos_of(sw) == SPC - 1) || sl_one;        // the segment at sw ends its chunk

  // ---- LD: chunk requests (address ic, cleft left), the chunk buffer, segment delivery
  logic [WA-1:0] ic;
  logic [31:0]   cleft;
  // cleft != 0 and cleft == 1, registered with cleft: no 32-bit compare on b_req's path into
  // otpu_native_dram's queue (the build's DMA -> memory path)
  logic          cl_nz, cl_one;
  localparam bit H16 = ST16 != 0;                     // (16-bit matrices)
  // r * n < lim (lim <= 2 * SPC; H16 4 * SPC), without an r x n multiply: both factors are below
  // lim then, so their product on clog2(lim) bits each decides
  localparam int PLM = (1 << $clog2((H16 ? 4 : 2) * SPC)) - 1;
  function automatic logic prod_lt(input logic [8:0] r, input logic [5:0] n, input int lim);
    if (r == 0 || n == 0) return 1'b1;
    if (32'(r) >= lim || 32'(n) >= lim) return 1'b0;
    return (32'(r) & PLM) * (32'(n) & PLM) < lim;
  endfunction
  // fp32 -> the 16-bit format (a stream's updated values as DRAM keeps them; opentpu.isasim
  // st16_pack, docs/isa.md "ST16"), to nearest even. e6m9 (ST16 1): 6 exponent bits (bias 31),
  // 9 mantissa bits; below 2^-30 (exponent < 97) a signed zero, a finite magnitude past the
  // largest ((2 - 2^-9) 2^31, or rounding up to 2^32) saturates to it, an infinity stays one, a
  // NaN a NaN (its quiet bit). bf16 (ST16 2): the fp32's high half rounded, a subnormal a
  // signed zero, an infinity one, a NaN a NaN (its quiet bit set)
  function automatic logic [15:0] st16_pack(input logic [31:0] x);
    logic [14:0] r;
    if (ST16 == 2) begin
      if (x[30:23] == 8'h00) return {x[31], 15'd0};
      if (x[30:23] == 8'hFF) return x[31:16] | {9'd0, |x[22:0], 6'd0};
      return x[31:16] + 16'(x[15] && (x[14:0] != 0 || x[16]));
    end
    if (x[30:23] == 8'hFF) return {x[31], 6'h3F, |x[22:0], 8'd0};
    if (x[30:23] < 8'd97) return {x[31], 15'd0};
    if (x[30:23] > 8'd158) return {x[31], 15'h7DFF};
    r = {6'(x[30:23] - 8'd96), x[22:14]} + 15'(x[13] && (x[12:0] != 0 || x[14]));
    return {x[31], r[14:9] == 6'h3F ? 15'h7DFF : r};
  endfunction
  // the 16-bit format -> fp32 (a stream's values as they are fed; opentpu.isasim st16_f32)
  function automatic logic [31:0] st16_x(input logic [15:0] h);
    if (ST16 == 2) return {h, 16'd0};
    if (h[14:9] == 6'h00) return {h[15], 31'd0};
    if (h[14:9] == 6'h3F) return {h[15], 8'hFF, h[8:0], 14'd0};
    return {h[15], 8'(h[14:9]) + 8'd96, h[8:0], 14'd0};
  endfunction
  logic [PW:0]   occ, cnt;             // chunks requested / received, and not yet delivered
  logic [PW-1:0] wp, rp;               // buffer slot of the next chunk received / delivered
  logic ds_wreq, ds_eat;                                    // DSTEP: a chunk write; a chunk used
  wire ld_act = busy && !is_st && !ackw;
  localparam int RUN = 64;                            // DSTEP: chunks per DRAM read / write run
  localparam int SRUN = 32;                           // ST: chunks per write run (<= DEPTH)
  initial if (SRUN > DEPTH) $fatal(1, "otpu_dma: SRUN > DEPTH");
  logic ds_rr;                                        // DSTEP: inside a read run
  logic [$clog2(RUN)-1:0] ds_rc;                      // ... its chunks issued
  // occ != DEPTH and occ <= DEPTH - RUN, registered with occ (oc_nf, oc_lr): b_req's compares of
  // occ ran on into the port B grant, the MXU's request and otpu_native_dram's queues (the
  // 959b425 build's DMA -> memory paths started at occ). occ moves by at most one a cycle, so the
  // next flags pick among compares of occ - 1, occ and occ + 1 by the late increment / decrement
  logic          oc_nf, oc_lr;
  function automatic logic [1:0] oc_of(input logic [PW:0] o);
    return {o != (PW+1)'(DEPTH), o <= (PW+1)'(DEPTH - RUN)};
  endfunction
  function automatic logic [1:0] oc_nx(input logic [PW:0] o, input logic inc, input logic dec);
    return (inc && !dec) ? oc_of(o + 1'b1) : (dec && !inc) ? oc_of(o - 1'b1) : oc_of(o);
  endfunction
`ifndef SYNTHESIS
  // the checks start once the reset has held two edges (registers start arbitrary, and so does a
  // unit's reset register until the slice's reset pipeline fills: otpu_slice)
  bit rst_p, rst_seen;
  initial begin rst_p = 1'b0; rst_seen = 1'b0; end
  always @(posedge clk) begin rst_p <= rst; if (rst && rst_p) rst_seen <= 1'b1; end
  always_ff @(posedge clk)
    if (!rst && rst_seen && busy && (cl_nz != (cleft != 0) || cl_one != (cleft == 1)))
      $fatal(1, "otpu_dma: cl_nz %0d cl_one %0d but cleft %0d", cl_nz, cl_one, cleft);
  always_ff @(posedge clk)
    if (!rst && rst_seen && busy && {oc_nf, oc_lr} != oc_of(occ))
      $fatal(1, "otpu_dma: oc_nf %0d oc_lr %0d but occ %0d", oc_nf, oc_lr, occ);
  // the buffer: no more than DEPTH chunks requested (an ST's: pushed) and not yet delivered, and
  // no more received than requested (read data cannot be refused: its slot is reserved)
  always_ff @(posedge clk)
    if (!rst && rst_seen && busy && (occ > (PW+1)'(DEPTH) || cnt > occ))
      $fatal(1, "otpu_dma: %0d chunks reserved, %0d received, in a buffer of %0d", occ, cnt, DEPTH);
  always_ff @(posedge clk)
    if (!rst && rst_seen && busy && (sl_nz != (sleft != 0) || sl_one != (sleft == 1)))
      $fatal(1, "otpu_dma: sl_nz %0d sl_one %0d but sleft %0d", sl_nz, sl_one, sleft);
`endif
  wire fr_act = LDQ > 1 && sl_f;      // LDQ 2: the front's requests (under an LD: with its own)
  wire ld_req = (ld_act || (fr_act && is_st)) && cl_nz && oc_nf &&
                (!is_ds || ds_rr || (!ds_wreq && oc_lr));
  wire ld_iss = ld_req && b_gnt && !ds_wreq;
  // LDA (above): the delivery's row la_r (of la_rl + 1) and the next row's first word la_nx; the
  // requests' row lq_r (of lq_rl + 1, lq_nw words a row lq_rs apart: their own, an LDA in the
  // front requests under the back's delivery; rq_la: the requests are an LDA's) and next first word
  // lq_nx (lq_nc: that row's chunks); a segment's phases
  // la_ph (an ACT write each: LBP for bytes, LRP for records), its index in the row la_sc (of
  // la_nbs byte segments, then the records: la_rec); aq_n the ACT writes taken and not yet out
  // of the queue (la_ok: room for one more)
  localparam int LBP = (4 * W) / ((ALANES > 0) ? ALANES : 1);
  localparam int LRP = W / 2;
  localparam int AQD = 4;
  logic          is_la, la_rec, la_dr, rq_la;
  logic [3:0]    la_ph;
  logic [7:0]    la_r, lq_r, la_rl, lq_rl, la_ab, la_ar;
  logic [15:0]   la_nw, lq_nw, la_ns, la_sc, la_nbs;
  logic [WA-1:0] la_rs, lq_rs, la_nx, lq_nx;
  logic [2:0]    aq_n;
  wire [31:0]    lq_nc = (32'(lq_nw) + 32'(lq_nx % CW) + (CW - 1)) >> CWL;
  wire           la_ok = aq_n < 3'(AQD);
  wire           la_end = la_ph == 4'(la_rec ? LRP - 1 : LBP - 1);
  wire           la_v = LDA && is_la && ld_act && sl_nz && (cnt != 0) && la_ok;
  // deliver the segment at sw (LDA: at its last phase)
  wire ld_dv  = ld_act && sl_nz && (cnt != 0) && (!LDA || !is_la || (la_ok && la_end));
  wire ld_eat = (ld_dv && seg_end) || ds_eat;               // ... which frees its chunk's slot

  // block RAM (a technology RAM, rtl/tech), read every cycle
  logic [D*8-1:0] lb_q;
  // ST: a complete chunk (st_push: its data st_cd, word mask st_cm) into the buffer at wp;
  // a write run (st_wq) issues the chunk at rp, and the buffer's read follows the next one
  logic           st_push, st_iss, st_wq;
  logic [D*8-1:0] st_cd;
  logic [CW-1:0]  st_cm;
  (* ram_style = "distributed" *) logic [CW-1:0] lm [DEPTH];
  otpu_ram_1r1w #(.W(D*8), .N(DEPTH), .STYLE("block")) u_lb (
    .clk, .we(b_rvalid || st_push), .wa(wp), .wd(b_rvalid ? b_rdata : st_cd), .re(1'b1),
    .ra(st_iss ? rp + 1'b1 : rp), .rd(lb_q));
  always_ff @(posedge clk)
    if (st_push) lm[wp] <= st_cm;
  // ---- streams (DSTEP, STREAM; docs/stream.md)
  localparam int NGC = 2 * RUN;                      // output gather (chunks)
  localparam int NG = NGC * SPC;                     // ... in segments
  localparam int GW = $clog2(NG) + 1;
  localparam int CBD = 256 / W;                      // row output buffer per lane
  // SE's tail is built for 8 lanes (its isum_64 partial loops need 64 / W > the adder's
  // latency); other widths have no streams (the compiler emits the VOP sequence)
  localparam bit HAS_SS = HAS_DSTEP && W == 8 && LANES == 8;
  // SR > 1 (the header): a row step's rows and their chunks
  localparam int SRW = $clog2(SR);
  localparam bit SRX = SR > 1;
  initial if (SR < 1 || SR != (1 << SRW) || SR > SPC || (SRX && !HAS_SS))
    $fatal(1, "otpu_dma: SR %0d (a power of two up to D / 4 / W = %0d, with streams)", SR, SPC);
  logic [31:0]  ds_oa;                               // o (TMEM)
  logic [WA-1:0] ds_wb;                              // the stream's write (DRAM words)
  // the stream's next chunk write, ds_wb + ds_wch * CW, as a register: the adder ran into port B's
  // address and its channel parity (b_par), the DMA -> memory path left after 23ee14c
  logic [WA-1:0] ds_wa;
  // pad64 (SE's one tree, cols = 64): a row is its 8 segments and 8 of +0 (not read), and the Y of
  // those is dropped (docs/stream.md 11.3); ds_sj / ds_yj: the segment fed / out in its row
  logic         ds_pad;
  logic [3:0]   ds_sj, ds_yj;
  wire          ds_spad = ds_pad && ds_sj[3];         // the segment to feed is padding
  logic [8:0]   ds_rows;
  logic         ds_qen;                              // the stream has row outputs o
  logic [15:0]  ds_nseg, ds_left, ds_ycnt;           // segments: all, still to feed, out of SE
  logic [8:0]   ds_ocnt;                             // o values out of SE
  logic [15:0]  ds_wch;                              // chunks written
  logic [7:0]   ds_nv;                               // o segments (rows / W, rounded up)
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && rst_seen && !SRX && busy && is_ds && ds_wreq && ds_wa != ds_wb + 32'(ds_wch) * CW)
      $fatal(1, "otpu_dma: ds_wa %h but ds_wb %h + %0d chunks", ds_wa, ds_wb, ds_wch);
`endif
  // before the run: a STREAM's descriptor read (ds_dsc) and capture (ds_dsd), the setup (ds_su;
  // a DSTEP's from its command), SE's grant (ds_wait), the fill (ds_fill)
  cmd_t         sc;                                  // the DSTEP / STREAM
  logic         ds_dsc, ds_dsd, ds_su, ds_wait;
  logic [31:0]  dsw [8];                             // the STREAM's descriptor words
  // TMEM's input stage (release r2, docs/tmem_nocopy.md 7): a read's data comes two cycles after
  // its request forms (the request register, then the RAM), so the fill's tags pass fr_k0 and
  // the run starts a cycle after the fill's last request (ds_fx), and a STREAM's descriptor is
  // captured a cycle after its request leaves the register (ds_dw)
  logic [2:0]   fr_k0, fr_k;                         // the fill step requested / read last cycle
  logic [4:0]   fr_i0, fr_i;
  logic         ds_fx, ds_dw;
  // H16: a 16-bit chunk's 2 SPC segments (ds_hr / ds_hw: the stream's source / destination is
  // in 16 bits)
  localparam int SPB = H16 ? 2 * SPC : SPC;
  logic [$clog2(SPB > 1 ? SPB : 2)-1:0] ds_pos, pe_pos;   // segment within the chunk at rp
  logic         ds_hr, ds_hw;
  wire          ds_pend = 32'(ds_pos) == (ds_hr ? 2 * SPC - 1 : SPC - 1);  // ds_pos ends its chunk
  (* max_fanout = 64 *) logic pe;
  logic         pe_in, pe_zero;
  logic [GW-1:0] og;                                 // gathered updated segments (<= NG)
  logic [$clog2(NG)-1:0] gt;                         // next gather slot
  (* max_fanout = 64 *) logic [$clog2(NGC)-1:0] gh;  // the chunk to write next
  logic [W*32-1:0] y_p, gq [SPC];                   // gq[p]: segment p of chunk gh
  logic         y_v, o_v;
  // Simulation only (docs/se_engine.md, tools/vpu_hack.py nostate): with +ds_sim=1 a DSTEP
  // with flag bit 7 (DF_SIM_NOWR) discards its updated state, no DRAM writes (the stream's
  // counters count them as today, so it ends as today): with ZERO (no reads), the timing of a
  // step whose state traffic costs nothing. Synthesis ties it off
  logic         ds_nowr;
  // ... and with +ds_beside=1 every stream runs on a phantom engine here instead of SE: SE is never
  // requested (the VPU keeps running VOPs) and the updated segments and row outputs come out of
  // delay lines on pe (PH_Y, PH_O pe cycles after their segment; their data is +0): the timing of
  // a stream engine of its own beside the VPU (tools/vpu_hack.py nostate, docs/se_engine.md)
`ifndef SYNTHESIS
  bit           ds_sim, ds_bes;
  initial ds_sim = $test$plusargs("ds_sim=1");
  initial ds_bes = $test$plusargs("ds_beside=1");
`else
  wire          ds_sim = 1'b0, ds_bes = 1'b0;
`endif
  wire          y_keep = y_v && !(ds_pad && ds_yj[3]) && !ds_nowr;   // a Y segment kept
  // SR = 1: a kept segment lands in the gather (16-bit: with the one before it, yh; y_odd)
  logic         y_odd;
  wire          y_land = y_keep && (!ds_hw || y_odd);
  logic [31:0]  y_d [SR*W], in_d [SR*W];
  // 16-bit: the updated values packed (yk), the even segment's held (yh, a row group's W), and
  // the words written: y_d's, or the held segment's pairs (lanes < W / 2) and this one's
  logic [15:0]  yk [SR*W];
  logic [W*16-1:0] yh [SR];
  logic [31:0]  yw [SR*W];
  logic [SR-1:0][31:0] o_d;
  // the row outputs: o value i in entry i / W of lane i % W's buffer, a LUT RAM (ob_q: the
  // entries at oi, read for the TMEM write). An o value lands a cycle after it leaves the
  // datapath: its lane's enable, the entry and the value, registered. As 256 x 32 flip-flops
  // with one-hot entry enables (o_d went to all 256 entries, their enables decoded from
  // ds_ocnt: 8 ns of wire at MCOLS=4) the value's and the enables' wires to the entries were
  // the worst core paths of the fused 133.33 MHz build c2830d6 (+0.092 ns, no logic). ds_out,
  // the first read of ob, starts on the edge that writes the last o at the earliest
  localparam int OBW = $clog2(CBD);
  logic [W-1:0]    ow_e;                             // lane l takes o_q ...
  (* keep *) logic [OBW-1:0] ow_a [W];               // ... at entry ow_a[l] (a copy per lane)
  logic [SR-1:0][31:0] o_q;                          // (SR > 1: lane l takes row group l % SR's)
  logic [31:0]     ob_q [W];
  logic [7:0]   oi;                                  // o segment written to TMEM next
  // SE runs up to SE_LAG cycles behind pe (otpu_vpu registers the ss_* inputs) and qualifies
  // y_v and o_v with its own pe: taking one more segment needs room for it, this cycle's y and
  // the segments of the last SE_LAG pe cycles (pe's counted, the older ones as always taken)
  localparam int SE_LAG = 2;
  // (SR > 1: the room is counted in row steps of SR segments, less the updated-segment
  // registers' 2 SR SPC; the current set's chunks instead of the buffer's head; o counts row
  // steps; a chunk leaves the buffer as it goes into the next set, sr_eat)
  logic sr_cv, sr_eat, sr_idle;
  logic [8:0] ds_nr;                                 // SR > 1: row steps
  wire  ds_room = SRX ?
      ({1'b0, og} + (GW+1)'(SR * y_v) + (GW+1)'(SR * pe)) <=
          (GW+1)'(NG - SR * SE_LAG - 2 * SR * SPC) :
      ({1'b0, og} + (GW+1)'(y_v) + (GW+1)'(pe)) <= (GW+1)'(NG - SE_LAG);
  wire  ds_take = ds_run && (ds_left != 0) && (ds_zero || ds_spad || (SRX ? sr_cv : cnt != 0))
                  && ds_room;
  wire  ds_flushed = (ds_ycnt == ds_nseg) && (!ds_qen || ds_ocnt == (SRX ? ds_nr : ds_rows));
  wire  ds_drain = ds_run && (ds_left == 0) && !ds_flushed && ds_room;
  assign ds_eat = SRX ? sr_eat : ds_take && !ds_zero && !ds_spad && ds_pend;
  // a registered request (it selects the 1024-bit write data: replicated, no decode on the
  // path); it follows og, which only counts during a stream
  (* max_fanout = 64 *) logic ds_wr;
  logic sr_gl;                                       // SR > 1: a chunk lands in the gather
  wire  [GW-1:0] og_nx = og + (SRX ? (sr_gl ? GW'(SPC) : GW'(0)) : GW'(y_land)) -
                         ((ds_wreq && b_gnt) ? GW'(SPC) : GW'(0));
  wire  ds_rr_nx = ds_rr && !(ld_iss && (32'(ds_rc) == RUN - 1 || cl_one));
  // a write run: once RUN chunks are in (or the stream has ended), then while chunks are in
  wire  ds_wr_nx = (og_nx >= GW'(SPC)) &&
                   (ds_wr || (!ds_rr_nx && (og_nx >= GW'(RUN * SPC) || ds_left == 0)));
  assign ds_wreq = ds_wr;
  // the gather: one simple dual-port RAM per segment position and lane (write: word l of the
  // segment from SE at chunk gt / SPC; read: chunk gh). The write is registered here (the
  // segment, and per RAM a kept copy of its enable and chunk), so it lands an edge after SE
  // hands it over: from SE's registers straight into the RAMs' data inputs and gt's fan-out to
  // their write addresses, it was 0 levels at +0.228 ns on the fused 133.33 MHz build c2830d6.
  // og counts the segment as before, so a write run may read the chunk in the cycle the
  // segment is still in the register: that cycle takes it from there (gb_by, from registers:
  // the in-flight segment's enable and whether its chunk is the one gh points at, gb_eq,
  // registered from gh's next value, so b_wdata stays one LUT after the RAM). Only a chunk's
  // last segment can be read in flight (a run reads complete chunks), so only position
  // SPC - 1 has the bypass
  // The RAMs read ahead (otpu_ram_1r1w_ra: block RAM, read at gh's next value, gh_n, with its
  // own bypass of the word written at the same edge): as LUT RAMs every address bit of gh and of
  // the write reached all of their ~3,500 cells (the card's m8: 39,424 address pins, the most of
  // any memory), with a kept copy of the write's enable and chunk per RAM; on block RAMs gh_n
  // reaches one per position and lane, and the write one register per position
  localparam int GCW = $clog2(NGC);
  logic [W*32-1:0] y_q;                               // the segment in flight
  logic           gb_we [SPC];                        // it lands in position p's RAMs ...
  logic [GCW-1:0] gb_wa [SPC];                        // ... at this chunk
  (* keep *) logic           gb_eq [W];               // the bypass (position SPC - 1)
  logic [GCW-1:0] gh_n;                               // gh's next value (the read ahead)
  // SR > 1: a whole chunk lands (sr_gw: the chunk's segments, sr_gt: its slot; sr_gl the
  // registered write, which og counts as it lands: no read of a chunk in flight, no bypass)
  logic [W*32-1:0] sr_gq [SPC];
  logic            sr_gw;
  logic [GCW-1:0]  sr_gt;
  for (genvar p = 0; p < SPC; p++) begin : g_gb
    for (genvar l = 0; l < W; l++) begin : g_l
      logic [31:0] gr;                                // gb[gh]
      otpu_ram_1r1w_ra #(.W(32), .N(NGC), .STYLE("block")) u_gb (
        .clk, .we(gb_we[p]), .wa(gb_wa[p]), .wd(SRX ? sr_gq[p][32 * l +: 32] : y_q[32 * l +: 32]),
        .ra_n(gh_n), .rd(gr));
      wire gb_by = !SRX && p == SPC - 1 && gb_we[p] && gb_eq[l];   // chunk gh's segment p is in y_q
      assign gq[p][32 * l +: 32] = gb_by ? y_q[32 * l +: 32] : gr;
`ifndef SYNTHESIS
      // the bypass is taken exactly when a write run reads the segment still in flight
      always_ff @(posedge clk)
        if (!rst && rst_seen && ds_wreq && gb_by != (gb_we[p] && gb_wa[p] == gh))
          $fatal(1, "otpu_dma: gather bypass %0b for chunk %0d segment %0d", gb_by, gh, p);
`endif
    end
  end
  always_ff @(posedge clk) begin
    y_q <= y_p;
    for (int p = 0; p < SPC; p++) begin
      gb_we[p] <= SRX ? sr_gw : y_land && 32'(gt) % SPC == p;
      gb_wa[p] <= SRX ? sr_gt : GCW'(32'(gt) / SPC);
    end
    for (int l = 0; l < W; l++)
      gb_eq[l] <= GCW'(32'(gt) / SPC) == ((ds_wreq && b_gnt) ? GCW'(gh + 1'b1) : gh);
    sr_gl <= !rst && SRX && sr_gw;
  end
  // gh's next value as the state register's block below moves it: a DSTEP's or a STREAM's start
  // clears it, a stream's taken chunk write advances it (reset leaves it)
  assign gh_n = rst ? gh :
                (go && (gc.op == OP_DSTEP || gc.op == OP_STREAM)) ? '0 :
                (!go && !ackw && busy && is_ds && ds_wreq && b_gnt) ? GCW'(gh + 1'b1) : gh;
`ifndef SYNTHESIS
  logic [GCW-1:0] gh_q;
  bit             gh_qv = 1'b0;
  always_ff @(posedge clk) begin
    gh_q <= gh_n;
    gh_qv <= 1'b1;
    if (gh_qv && gh != gh_q) $fatal(1, "otpu_dma: gh is %0d, not gh_n's %0d", gh, gh_q);
  end
`endif
  logic [31:0] sr_in [SR*W];                         // SR > 1: the segments fed (pe_* there)
  logic [W*32-1:0] in_w;                             // SR = 1: the fed position's W words
  always_comb begin
    for (int i = 0; i < SR * W; i++) yk[i] = st16_pack(y_d[i]);
    for (int g = 0; g < SR; g++)
      for (int l = 0; l < W; l++)
        yw[g * W + l] = !ds_hw ? y_d[g * W + l] :
                        (l < W / 2) ? yh[g][32 * l +: 32] :
                                      {yk[g * W + (2 * l + 1) % W], yk[g * W + (2 * l) % W]};
    // (16-bit: segment pe_pos's values are in the words of position pe_pos / 2, its lane l's
    // half l % 2 of word l / 2 there, after the W / 2 words of the even segment before it)
    in_w = lb_q[W * 32 * (ds_hr ? 32'(pe_pos) / 2 : 32'(pe_pos) % SPC) +: W * 32];
    for (int l = 0; l < W; l++) begin
      y_p[32 * l +: 32] = yw[l];
      if (SRX)
        for (int g = 0; g < SR; g++) in_d[g * W + l] = sr_in[g * W + l];
      else
        in_d[l] = pe_zero ? 32'd0 :
                  !ds_hr ? in_w[32 * l +: 32] :
                          st16_x(in_w[32 * ((32'(pe_pos) % 2) * (W / 2) + l / 2) + 16 * (l % 2) +: 16]);
    end
  end
  always_ff @(posedge clk) begin
    if (y_keep && ds_hw)
      for (int i = 0; i < SR * W; i++) yh[i / W][16 * (i % W) +: 16] <= yk[i];
    if (rst || go) y_odd <= 1'b0;
    else if (y_keep && ds_hw) y_odd <= !y_odd;
  end

  // ---- SR > 1: row steps (the header). The reads and the writes step per row step i, chunk
  // column j, row g (sr_icn, sr_wan: the next request's address); a chunk from the buffer's head
  // goes into the next set (two banks of SR chunk registers: sel's is the current set, fed; the
  // other the next, loaded in arrival order), which becomes the current one as that is used up;
  // the updated segments collect into one chunk register per row (two banks: ysel's fills, the
  // other drains into the gather a chunk a cycle: sr_gw, sr_gt, sr_gq)
  logic [WA-1:0] sr_icn, sr_wan;
  if (SRX) begin : g_sr
    localparam int RB = SRW + 1;                      // a count of rows, 0 .. SR
    logic [3:0]    cpr_r, cpr_w;                      // chunks a row (read, written)
    logic [RB-1:0] np;                                // rows of the last row step
    logic [15:0]   rcw_r, bcw_r, rcw_w, bcw_w;        // a row's and a row step's words (each side)
    always_ff @(posedge clk)
      if (busy && is_ds && ds_su) begin
        cpr_r <= 4'(su_ns / (ds_hr ? 2 * SPC : SPC));
        cpr_w <= 4'(su_ns / (ds_hw ? 2 * SPC : SPC));
        np  <= RB'((32'(su_cfg.rows) - 1) % SR + 1);
        rcw_r <= 16'(32'(su_ns) / (ds_hr ? 2 * SPC : SPC) * CW);
        bcw_r <= 16'(32'(su_ns) / (ds_hr ? 2 * SPC : SPC) * CW * SR);
        rcw_w <= 16'(32'(su_ns) / (ds_hw ? 2 * SPC : SPC) * CW);
        bcw_w <= 16'(32'(su_ns) / (ds_hw ? 2 * SPC : SPC) * CW * SR);
      end
    function automatic logic [RB-1:0] rows_of(input logic [8:0] i, input logic [8:0] nr,
                                              input logic [RB-1:0] last);
      return (i + 1'b1 == nr) ? last : RB'(SR);          // row step i's rows
    endfunction
`ifndef SYNTHESIS
    always_ff @(posedge clk)
      if (!rst && busy && is_ds && ds_su && 32'(su_ns) % ((ds_hr || ds_hw) ? 2 * SPC : SPC) != 0)
        $fatal(1, "otpu_dma: SR %0d: %0d segments a row, not whole chunks", SR, su_ns);
`endif

    // the reads' addresses: row step rq_i (its first word rq_b), column rq_j, row rq_g
    logic [8:0]    rq_i;
    logic [3:0]    rq_j;
    logic [RB-1:0] rq_g;
    logic [WA-1:0] rq_b;
    wire  [RB-1:0] rq_gm = rows_of(rq_i, ds_nr, np) - 1'b1;
    always_comb
      if (rq_g != rq_gm)             sr_icn = ic + 32'(rcw_r);
      else if (rq_j + 1'b1 != cpr_r) sr_icn = rq_b + 32'(rq_j + 1'b1) * CW;
      else                           sr_icn = rq_b + 32'(bcw_r);
    always_ff @(posedge clk)
      if (go) begin
        rq_i <= '0; rq_j <= '0; rq_g <= '0;
      end else if (busy && is_ds && ds_su) begin
        rq_b <= su_src >> 2;
      end else if (busy && is_ds && ld_iss) begin
        if (rq_g != rq_gm) rq_g <= rq_g + 1'b1;
        else begin
          rq_g <= '0;
          if (rq_j + 1'b1 != cpr_r) rq_j <= rq_j + 1'b1;
          else begin
            rq_j <= '0; rq_i <= rq_i + 1'b1; rq_b <= rq_b + 32'(bcw_r);
          end
        end
      end

    // the sets: a chunk from the buffer's head into the next set (cl_go; it is lb_q a cycle
    // later, as cl_q lands it), the next set the current one once that is used up or empty
    // (flip; a chunk landing then lands in the set that becomes current)
    logic [D*8-1:0] sb [2][SR];
    logic           sel, cv, cl_q;
    logic [RB-1:0]  cnp;                              // the current set's rows
    logic [SRW-1:0] cl_g;                             // the landing chunk's row
    logic [8:0]     ls_i;                             // the next set's row step
    logic [3:0]     ls_j;                             // ... column
    logic [RB-1:0]  ls_n;                             // ... rows taken
    wire  [RB-1:0]  ls_np = rows_of(ls_i, ds_nr, np);
    wire            ls_full = ls_n == ls_np;
    wire            cl_go = busy && is_ds && !ds_zero && !ls_full && cnt != 0;
    wire            used = ds_take && !ds_zero && !ds_spad && ds_pend;
    wire            flip = (!cv || used) && ls_full;
    assign sr_eat = cl_go;
    assign sr_cv = cv;
    always_ff @(posedge clk) begin
      cl_q <= !rst && cl_go;
      cl_g <= SRW'(ls_n);
      if (cl_q) sb[!sel][cl_g] <= lb_q;
      if (rst || go) begin
        sel <= 1'b0; cv <= 1'b0; ls_i <= '0; ls_j <= '0; ls_n <= '0;
      end else begin
        if (cl_go) ls_n <= ls_n + 1'b1;
        if (used) cv <= 1'b0;
        if (flip) begin
          sel <= !sel; cv <= 1'b1; cnp <= ls_np; ls_n <= '0;
          if (ls_j + 1'b1 != cpr_r) ls_j <= ls_j + 1'b1;
          else begin
            ls_j <= '0; ls_i <= ls_i + 1'b1;
          end
        end
      end
    end
    // the segments fed: the current set's at ds_pos, registered as pe; +0 for ZERO, pad64's
    // padding and the rows past the last
    logic [31:0]   fd_q [SR*W];
    logic          fz_q;
    logic [SR-1:0] gz_q;
    logic [W*32-1:0] fd_w [SR];                       // the fed position's W words, a row group's
    always_comb
      for (int g = 0; g < SR; g++)
        fd_w[g] = sb[sel][g][W * 32 * (ds_hr ? 32'(ds_pos) / 2 : 32'(ds_pos) % SPC) +: W * 32];
    always_ff @(posedge clk) begin
      for (int g = 0; g < SR; g++) begin
        for (int l = 0; l < W; l++)        // (16-bit: as in_d, at ds_pos)
          fd_q[g * W + l] <= !ds_hr ? fd_w[g][32 * l +: 32] :
                             st16_x(fd_w[g][32 * ((32'(ds_pos) % 2) * (W / 2) + l / 2) +
                                            16 * (l % 2) +: 16]);
        gz_q[g] <= RB'(g) >= cnp;
      end
      fz_q <= ds_zero || ds_spad;
    end
    for (genvar i = 0; i < SR * W; i++) begin : g_in
      assign sr_in[i] = (fz_q || gz_q[i / W]) ? 32'd0 : fd_q[i];
    end

    // the updated chunks: segment yp of row step yi's column yj (the kept segments: y_keep);
    // a full bank flips and drains its rows dg .. dn - 1
    logic [D*8-1:0] yb [2][SR];
    logic           ysel;
    logic [$clog2(SPB)-1:0] yp;
    logic [3:0]     yj;
    logic [8:0]     yi;
    logic [RB-1:0]  dn, dg;
    wire            y_end = y_keep && 32'(yp) == (ds_hw ? 2 * SPC - 1 : SPC - 1);
    // the words' position in the chunk register: yp, or (16-bit) yp / 2 after an odd segment
    wire            y_wr = y_keep && (!ds_hw || yp[0]);
    wire [$clog2(SPC)-1:0] yq = $clog2(SPC)'(ds_hw ? 32'(yp) / 2 : 32'(yp));
    always_ff @(posedge clk) begin
      if (y_wr)
        for (int g = 0; g < SR; g++)
          for (int l = 0; l < W; l++) yb[ysel][g][32 * (32'(yq) * W + l) +: 32] <= yw[g * W + l];
      for (int q = 0; q < SPC; q++) sr_gq[q] <= yb[!ysel][SRW'(dg)][W * 32 * q +: W * 32];
      if (rst || go) begin
        ysel <= 1'b0; yp <= '0; yj <= '0; yi <= '0; dn <= '0; dg <= '0; sr_gt <= '0;
      end else begin
        if (y_keep) yp <= y_end ? '0 : yp + 1'b1;
        if (y_end) begin
          ysel <= !ysel; dn <= rows_of(yi, ds_nr, np); dg <= '0;
          if (yj + 1'b1 != cpr_w) yj <= yj + 1'b1;
          else begin
            yj <= '0; yi <= yi + 1'b1;
          end
        end else if (dg != dn) dg <= dg + 1'b1;
        if (sr_gw) sr_gt <= sr_gt + 1'b1;
      end
    end
    assign sr_gw = busy && is_ds && dg != dn;
    assign sr_idle = dg == dn && !sr_gl;
`ifndef SYNTHESIS
    // a bank drains in SR <= SPC cycles, before the other one fills
    always_ff @(posedge clk)
      if (!rst && busy && is_ds && y_end && dg != dn && dg + 1'b1 != dn)
        $fatal(1, "otpu_dma: SR %0d: updated chunks %0d .. %0d not drained", SR, dg, dn);
`endif

    // the writes' addresses (the same order)
    logic [8:0]    wq_i;
    logic [3:0]    wq_j;
    logic [RB-1:0] wq_g;
    logic [WA-1:0] wq_b;
    wire  [RB-1:0] wq_gm = rows_of(wq_i, ds_nr, np) - 1'b1;
    always_comb
      if (wq_g != wq_gm)             sr_wan = ds_wa + 32'(rcw_w);
      else if (wq_j + 1'b1 != cpr_w) sr_wan = wq_b + 32'(wq_j + 1'b1) * CW;
      else                           sr_wan = wq_b + 32'(bcw_w);
    always_ff @(posedge clk)
      if (go) begin
        wq_i <= '0; wq_j <= '0; wq_g <= '0;
      end else if (busy && is_ds && ds_su) begin
        wq_b <= su_dst >> 2;
      end else if (busy && is_ds && ds_wreq && b_gnt) begin
        if (wq_g != wq_gm) wq_g <= wq_g + 1'b1;
        else begin
          wq_g <= '0;
          if (wq_j + 1'b1 != cpr_w) wq_j <= wq_j + 1'b1;
          else begin
            wq_j <= '0; wq_i <= wq_i + 1'b1; wq_b <= wq_b + 32'(bcw_w);
          end
        end
      end
  end else begin : g_sr1
    assign sr_icn = '0;
    assign sr_wan = '0;
    assign sr_eat = 1'b0;
    assign sr_cv = 1'b0;
    assign sr_idle = 1'b1;
    assign sr_gw = 1'b0;
    assign sr_gt = '0;
    for (genvar q = 0; q < SPC; q++) begin : g_gq
      assign sr_gq[q] = '0;
    end
    for (genvar i = 0; i < SR * W; i++) begin : g_in
      assign sr_in[i] = '0;
    end
  end

  // SE's ports (the tail and the reduction live in otpu_vpu)
  assign ss_pe = pe && !ds_bes;
  assign ss_in_v = pe_in && !ds_bes;
  assign ss_fk = ds_bes ? SF_NONE : fr_k;
  // the phantom engine (+ds_beside=1, simulation only): a segment's Y PH_Y pe cycles after it
  // was fed, a row's o PH_O after its last segment (about SE's own: the delay line, the trees)
  localparam int PH_Y = 120, PH_O = 150;
  logic [PH_O-1:0] ph_y, ph_o;
  logic [5:0]      ph_j;                             // the fed segment's index in its row
  wire  [5:0]      ph_ns = ds_pad ? 6'd16 : ss_cfg.ns;   // segments a row (pad64: 16)
  always_ff @(posedge clk)
    if (rst || !(busy && is_ds) || !ds_bes) begin
      ph_y <= '0; ph_o <= '0; ph_j <= '0;
    end else if (pe) begin
      ph_y <= {ph_y[PH_O-2:0], pe_in};
      ph_o <= {ph_o[PH_O-2:0], pe_in && ph_j == ph_ns - 1'b1};
      if (pe_in) ph_j <= (ph_j == ph_ns - 1'b1) ? '0 : ph_j + 1'b1;
    end
  assign ss_fi = fr_i;
  for (genvar l = 0; l < LANES; l++) begin : g_ss
    for (genvar g = 0; g < SR; g++) begin : g_r
      assign ss_in_d[g * LANES + l] = (l < W) ? in_d[g * W + l % W] : 32'd0;
    end
    assign ss_fd[l] = t_rdata[l];
  end
  if (HAS_SS) begin : g_ss_on
    assign y_v = ds_bes ? ph_y[PH_Y-1] && pe : ss_y_v;
    assign o_v = ds_bes ? ph_o[PH_O-1] && pe : ss_o_v;
    assign o_d = ss_o_d;
    for (genvar l = 0; l < SR * W; l++) begin : g_y
      assign y_d[l] = ss_y_d[l / W * LANES + l % W];
    end
  end else begin : g_nods
    assign y_v = 1'b0;
    assign o_v = 1'b0;
    assign o_d = '0;
    for (genvar l = 0; l < SR * W; l++) begin : g_y
      assign y_d[l] = '0;
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk)
      if (!rst && is_ds)
        $fatal(1, "otpu_dma: no streams here (W = %0d, LANES = %0d, HAS_DSTEP = %0d)", W, LANES,
               HAS_DSTEP);
`endif
  end

  // ---- a stream's setup (ds_su): a DSTEP's from its command (GDN); a STREAM's from its
  // command (w1 = desc | ks << 16) and its descriptor (docs/stream.md 3.2: float-safe words,
  // the payload is the sign bit and the mantissa), which must be in the hardware subset (4.4,
  // opentpu.isa.stream_hw_cfg): its scalar op list names dmode
  function automatic logic [23:0] dpl(input logic [31:0] w);
    return {w[31], w[22:0]};
  endfunction
  function automatic logic [23:0] dop(input int op, input int dst, input int a, input int b);
    return 24'(op | (dst << 4) | (a << 7) | (b << 11));
  endfunction
  localparam int SC_SUB = 1, SC_MUL = 2, O_A = 8, O_X = 9, O_K0 = 10, O_K1 = 11;
  ss_cfg_t     su_cfg;
  logic [AW-1:0] su_src, su_dst;         // DRAM byte addresses
  logic [31:0] su_vec, su_x, su_k, su_ks, su_out;
  logic [15:0] su_nseg;                  // segments SE takes (pad64: 16 per row)
  logic [5:0]  su_ns;                    // the stream's own segments per row (the fill's)
  logic        su_ok;
  always_comb begin
    logic [23:0] d0, d1, d2, d3, d4, o5, o6, o7;
    logic [11:0] rows, cols;
    logic [1:0]  dm, g;
    logic        dm_ok, g_ok, a_ok, fs_ok;
    d0 = dpl(dsw[0]); d1 = dpl(dsw[1]); d2 = dpl(dsw[2]); d3 = dpl(dsw[3]);
    d4 = dpl(dsw[4]); o5 = dpl(dsw[5]); o6 = dpl(dsw[6]); o7 = dpl(dsw[7]);
    // dmode from the op list (d2[3:0] ops)
    dm = SD_DELTA; dm_ok = 1'b1;
    if (d2[3:0] == 4'd3 && o5 == dop(SC_MUL, 0, O_A, O_K0) && o6 == dop(SC_SUB, 1, O_X, 0) &&
        o7 == dop(SC_MUL, 2, 1, O_K1))
      dm = SD_DELTA;
    else if (d2[3:0] == 4'd2 && o5 == dop(SC_SUB, 1, O_X, O_A) && o6 == dop(SC_MUL, 2, 1, O_K1))
      dm = SD_DELTA1;
    else if (d2[3:0] == 4'd1 && o5 == dop(SC_MUL, 2, O_X, O_K1))
      dm = SD_SCALE;
    else if (d2[3:0] == 4'd1 && o5 == dop(SC_MUL, 2, O_A, O_K1))
      dm = SD_DOT;
    else
      dm_ok = 1'b0;
    // G: a constant K0, slot 3 or one (g_src: 0 reg, 1 slot, 2 const, 3 one; g_idx)
    g = SG_K0; g_ok = 1'b1;
    if (d1[8:7] == 2'd2 && d1[11:9] == 3'd0) g = SG_K0;
    else if (d1[8:7] == 2'd1 && d1[11:9] == 3'd3) g = SG_COL;
    else if (d1[8:7] == 2'd3) g = SG_ONE;
    else g_ok = 1'b0;
    // A on slot 1 or 2 exactly when the op list reads it
    a_ok = (d1[0] == (dm != SD_SCALE)) && (!d1[0] || (d1[2:1] == 2'd0 &&
                                                       (d1[4:3] == 2'd1 || d1[4:3] == 2'd2)));
    fs_ok = 1'b1;
    for (int i = 0; i < 8; i++)
      if ((i < 5 || i < 5 + int'(d2[3:0])) && dsw[i][30:23] != 8'h80) fs_ok = 1'b0;
    rows = d0[11:0]; cols = d0[23:12];
    if (sc.op == OP_DSTEP) begin
      rows = 12'(sc.w4[15:0]); cols = 12'(sc.w4[31:16]);
    end
    su_cfg.ns = 6'(cols / W);
    su_cfg.rows = 9'(rows);
    su_cfg.a_en = d1[0];
    su_cfg.a_sel = d1[4:3] == 2'd2;
    su_cfg.dmode = dm;
    su_cfg.g_src = g;
    su_cfg.q_en = d1[20];
    su_src = AW'(cmd_a(sc, 2)); su_dst = AW'(cmd_a(sc, 3));
    su_vec = sc.w4; su_x = sc.w5; su_k = sc.w6;
    su_ks = 32'(sc.w1[31:16]); su_out = sc.w7;
    su_ok = fs_ok && dm_ok && g_ok && a_ok && rows != 0 && rows <= 12'd256 &&
            cols[5:0] == 0 && cols != 0 && cols <= 12'd256 && d4 == 0 &&
            d1[6:5] == 2'd1 && d1[13:12] == 2'd0 && d1[16:14] == 3'd1 && d1[19:17] == 3'd2 &&
            (!d1[20] || d1[22:21] == 2'd0) && !d1[23] && d2[13:4] == 0 &&
            !sc.flags[STF_SRC_T] && !sc.flags[STF_DST_T] && !sc.flags[STF_NODST];
    su_cfg.pair = 1'b0;
    if (sc.op == OP_DSTEP) begin
      su_cfg.a_en = 1'b1; su_cfg.a_sel = 1'b0; su_cfg.dmode = SD_DELTA; su_cfg.g_src = SG_K0;
      su_cfg.q_en = 1'b1; su_cfg.pair = DSTEP2 && sc.flags[DF_PAIR];
      su_src = AW'(cmd_a(sc, 1)); su_dst = AW'(cmd_a(sc, 1));
      su_vec = sc.w2; su_x = sc.w3; su_k = sc.w5;
      su_ks = sc.w7; su_out = sc.w6;
      su_ok = 1'b1;
    end
    su_ns = su_cfg.ns;
    su_cfg.pad64 = su_ns == 6'd8;
    if (su_cfg.pad64) su_cfg.ns = 6'd16;
    su_nseg = 16'(32'(SRX ? (32'(su_cfg.rows) + SR - 1) >> SRW : 32'(su_cfg.rows)) *
                  32'(su_cfg.ns));
  end
`ifndef SYNTHESIS
  // each end's bytes (a 16-bit end half the fp32's); one end alone 16-bit: not in place
  wire       su_hr = H16 && (sc.flags[STF_ST16] || ST16S && sc.flags[STF_ST16R]);
  wire       su_hw = H16 && (sc.flags[STF_ST16] || ST16S && sc.flags[STF_ST16W]);
  wire [31:0] su_sb = 32'(su_cfg.rows) * su_ns * W * (su_hr ? 2 : 4);
  wire [31:0] su_db = 32'(su_cfg.rows) * su_ns * W * (su_hw ? 2 : 4);
  always_ff @(posedge clk)
    if (!rst && rst_seen && ds_su && sc.op == OP_STREAM &&
        (!su_ok || su_src % D != 0 || su_dst % D != 0 ||
         ((su_dst != su_src || su_hr != su_hw) && su_dst < su_src + su_sb &&
          su_src < su_dst + su_db)))
      $fatal(1, "otpu_dma: STREAM's descriptor at %0d is not in the hardware subset", sc.w1[15:0]);
  always_ff @(posedge clk)
    if (!rst && ds_su && sc.op == OP_DSTEP && !DSTEP2 && sc.flags[DF_PAIR])
      $fatal(1, "otpu_dma: a DSTEP pair on a DMA without DSTEP2");
  always_ff @(posedge clk)
    if (!rst && ds_su && sc.op == OP_DSTEP && su_cfg.pair &&
        (su_cfg.rows[3:0] != 0 || 32'(su_ns) * W != 128))
      $fatal(1, "otpu_dma: DSTEP pair of %0d rows, %0d columns (rows a multiple of 16, 128)",
             su_cfg.rows, 32'(su_ns) * W);
`endif

  // the fill: one TMEM read of W words per cycle. Its step (fs_*: kind SF_*, index, address,
  // lanes) is registered and the next one computed a cycle ahead, so the counts stay off the
  // TMEM read address (clk125's DSTEP path at 125.49 MHz was counts -> compares -> address ->
  // TMEM bank address). The kinds go in the order q, k, a, g, x, K0, K1 (docs/stream.md 4.3),
  // skipping those the stream does not use (f_nk). A DSTEP pair's (DSTEP2) q, k, K0 and K1 take
  // twice the steps, head b's after head a's: at step f_kh the address jumps to head b's (f_kb:
  // its q | k 2 cols on with QKB, else head a's again; its K0 and K1 a word on).
  logic [2:0]  fk_k, fs_k;
  logic [4:0]  fk_i, fs_i;
  logic [31:0] fk_a, fs_a;
  logic [W-1:0] fk_m, fs_m;
  logic [2:0]  f_nk [8];                   // the kind after each kind (SF_NONE: the fill ends)
  logic [4:0]  f_kn [8];                   // each kind's steps less one
  logic [31:0] f_ka [8];                   // each kind's first address
  logic [4:0]  f_kh [8];                   // DSTEP2: the step after which head b's start
  logic [31:0] f_kb [8];                   // ... and their first address
  logic        f_pr;                       // ... the stream is a pair's
  function automatic logic [W-1:0] f_mask(input logic [2:0] k, input logic [4:0] i,
                                          input logic [8:0] rows);
    logic [W-1:0] m;
    for (int l = 0; l < W; l++)
      m[l] = (k == SF_X) ? ((32'(i) * W + l) < 32'(rows)) : ((k != SF_K0 && k != SF_K1) || l == 0);
    return m;
  endfunction
  wire fk_on = ds_fill && fs_k != SF_NONE;
  wire [2:0] f_nx = f_nk[fs_k];
  always_ff @(posedge clk) begin
    if (ds_su) begin
      logic [31:0] cw;
      logic [4:0]  nsm;
      cw = 32'(su_ns) * W;
      nsm = 5'(su_ns - 1'b1);
      f_nk[SF_NONE] <= SF_NONE;
      f_nk[SF_Q] <= SF_K;
      f_nk[SF_K] <= su_cfg.a_en && su_cfg.a_sel ? SF_A : su_cfg.g_src == SG_COL ? SF_G : SF_X;
      f_nk[SF_A] <= su_cfg.g_src == SG_COL ? SF_G : SF_X;
      f_nk[SF_G] <= SF_X;
      f_nk[SF_X] <= SF_K0;
      f_nk[SF_K0] <= SF_K1;
      f_nk[SF_K1] <= SF_NONE;
      f_kn[SF_NONE] <= '0;
      f_kn[SF_Q] <= nsm; f_kn[SF_K] <= nsm; f_kn[SF_A] <= nsm; f_kn[SF_G] <= nsm;
      f_kn[SF_X] <= 5'((32'(su_cfg.rows) + W - 1) / W - 1);
      f_kn[SF_K0] <= '0; f_kn[SF_K1] <= '0;
      f_ka[SF_NONE] <= '0;
      f_ka[SF_Q] <= su_vec; f_ka[SF_K] <= su_vec + cw; f_ka[SF_A] <= su_vec + 2 * cw;
      f_ka[SF_G] <= su_vec + 3 * cw; f_ka[SF_X] <= su_x;
      f_ka[SF_K0] <= su_k; f_ka[SF_K1] <= su_k + su_ks;
      fs_k <= su_cfg.q_en ? SF_Q : SF_K;
      fs_i <= '0;
      fs_a <= su_cfg.q_en ? su_vec : su_vec + cw;
      fs_m <= '1;
      f_pr <= su_cfg.pair;
      if (su_cfg.pair) begin
        logic [31:0] vb;
        vb = su_vec + (sc.flags[DF_QKB] ? 2 * cw : 32'd0);
        f_kn[SF_Q] <= 5'(2 * su_ns - 1); f_kn[SF_K] <= 5'(2 * su_ns - 1);
        f_kn[SF_K0] <= 5'd1; f_kn[SF_K1] <= 5'd1;
        f_kh[SF_Q] <= nsm; f_kh[SF_K] <= nsm; f_kh[SF_K0] <= '0; f_kh[SF_K1] <= '0;
        f_kb[SF_Q] <= vb; f_kb[SF_K] <= vb + cw; f_kb[SF_K0] <= su_k + 1;
        f_kb[SF_K1] <= su_k + su_ks + 1;
      end
    end else if (fk_on) begin
      if (fs_i == f_kn[fs_k]) begin
        fs_k <= f_nx; fs_i <= '0; fs_a <= f_ka[f_nx]; fs_m <= f_mask(f_nx, 5'd0, ds_rows);
      end else begin
        fs_i <= fs_i + 1'b1; fs_m <= f_mask(fs_k, fs_i + 1'b1, ds_rows);
        fs_a <= (DSTEP2 && f_pr && fs_i == f_kh[fs_k] &&
                 (fs_k == SF_Q || fs_k == SF_K || fs_k == SF_K0 || fs_k == SF_K1)) ?
                f_kb[fs_k] : fs_a + W;
      end
    end
    if (ds_dsd)
      for (int i = 0; i < 8; i++) dsw[i] <= t_rdata[i % LANES];
  end
  assign fk_k = fk_on ? fs_k : SF_NONE;
  assign fk_i = fk_on ? fs_i : 5'd0;
  assign fk_a = fk_on ? fs_a : 32'd0;
  assign fk_m = fk_on ? fs_m : '0;

  // ---- WAITW: the word of the chunk delivered (C1: ww_c1, ww_w), the compare (C2: ww_c2,
  // ww_ok), the TMEM write through lane 0 and the end, or the wait (ww_sl, ww_n) and the next
  // read; ww_tc counts the cycles against the timeout
  logic        is_ww, ww_c1, ww_c2, ww_ok, ww_sl, ww_late;
  logic [1:0]  ww_cmp;
  logic [WA-1:0] ww_a;
  logic [31:0] ww_t, ww_ref, ww_mask, ww_iv, ww_to, ww_n, ww_tc, ww_w;
  function automatic logic ww_holds(input logic [31:0] v, input logic [31:0] ref_,
                                    input logic [1:0] c);
    logic [31:0] dif;
    dif = v - ref_;
    case (c)
      2'd0:    return v == ref_;
      2'd1:    return v != ref_;
      default: return !dif[31];                    // GE: the difference >= 0, signed
    endcase
  endfunction

  // A WAITW's word held. The host wrote what the program reads after it before the word (the
  // WAITW's footprint orders every younger DRAM reader after it), through its own master, which
  // the board's DRAM adapter does not see: the adapter drops what it holds for port A (its reused
  // beat and runs: otpu_native_dram a_flush). One cycle, with ld_fin
  always_ff @(posedge clk) ww_held <= !rst && busy && is_ww && ww_c2 && ww_ok;

  // the delivered segment (in lb_q) goes to the TMEM write register; its position in the
  // chunk, lanes and TMEM address come with it (dv_*)
  logic                   dv_v, ld_last;
  logic [W-1:0]           dv_m;
  logic [31:0]            dv_a;
  int                     dv_p;
  logic [LANES-1:0]       lw_en;
  logic [LANES-1:0][31:0] lw_addr, lw_data;
  logic [W-1:0][31:0]     dv_seg;              // the delivered segment's words (lb_q at dv_p)
  always_comb for (int l = 0; l < W; l++) dv_seg[l] = lb_q[32 * (dv_p * W + l) +: 32];
  logic                   ds_ow;               // DSTEP: o segment oi goes to TMEM
  always_comb begin
    lw_en = '0; lw_addr = '0; lw_data = '0;
    for (int l = 0; l < W; l++) begin
      lw_en[l] = dv_v && dv_m[l] && !is_ww && !(LDA && is_la);   // (WAITW: after its compare)
      lw_addr[l] = dv_a + 32'(l);
      lw_data[l] = dv_seg[l];
      if (ds_ow) begin
        lw_en[l] = (32'(oi) * W + l) < 32'(ds_rows);
        lw_addr[l] = ds_oa + 32'(oi) * W + 32'(l);
        lw_data[l] = ob_q[l];
      end
    end
    if (ww_c2 && ww_ok) begin                       // WAITW (C3): the word that held
      lw_en[0] = 1'b1;
      lw_addr[0] = ww_t;
      lw_data[0] = ww_w;
    end
  end
  always_ff @(posedge clk) begin
    t_wen <= rst ? '0 : lw_en;
    t_waddr <= lw_addr;
    t_wdata <= lw_data;
  end

  // LDA: a phase's segment position, lanes, phase and row registered with it (the chunk is in
  // lb_q a cycle later), then its ACT write into the queue (aq: the bytes' ACT index, or the
  // record's block in idx; a record's enable in we[0], its {da, S} in d[63:0]); the queue's
  // head goes out (act_*, asc_*) in a cycle the quantizer has no write at its write stage
  if (LDA) begin : g_la
    localparam int AWD = ALANES / 4;                        // words an ACT byte write
    localparam int AQL = $clog2(AQD);
    typedef struct packed {
      logic                rec;
      logic [7:0]          row;
      logic [31:0]         idx;
      logic [ALANES-1:0]   we;
      logic [ALANES*8-1:0] d;
    } aq_t;
    (* ram_style = "distributed" *) aq_t aq [AQD];
    aq_t          ae, ah;
    logic [AQL:0] aq_w, aq_r;
    logic         la_v1, la_rec1;
    logic [3:0]   la_ph1;
    logic [W-1:0] la_m1;
    logic [15:0]  la_sc1;
    logic [7:0]   la_row1;
    always_ff @(posedge clk) begin
      la_v1 <= !rst && la_v;
      la_rec1 <= la_rec;
      la_ph1 <= la_ph;
      la_m1 <= sm;
      la_sc1 <= la_sc;
      la_row1 <= la_ar + la_r;
    end
    always_comb begin
      ae = '0;
      ae.rec = la_rec1;
      ae.row = la_row1;
      if (la_rec1) begin
        ae.idx = 32'(la_ab) + (32'(la_sc1 - la_nbs) * LRP) + 32'(la_ph1);
        ae.we[0] = la_m1[2 * la_ph1];
        ae.d[63:0] = {dv_seg[2 * la_ph1 + 1], dv_seg[2 * la_ph1]};
      end else begin
        ae.idx = 32'(la_ab) * D + 32'(la_sc1) * (4 * W) + 32'(la_ph1) * ALANES;
        for (int l = 0; l < ALANES; l++) begin
          ae.we[l] = la_m1[AWD * la_ph1 + l / 4];
          ae.d[8 * l +: 8] = dv_seg[AWD * la_ph1 + l / 4][8 * (l % 4) +: 8];
        end
      end
    end
    always_ff @(posedge clk) if (la_v1) aq[aq_w[AQL-1:0]] <= ae;
    assign ah = aq[aq_r[AQL-1:0]];
    wire aq_pop = (aq_w != aq_r) && !act_busy;
    always_ff @(posedge clk)
      if (rst) begin
        aq_w <= '0;
        aq_r <= '0;
        aq_n <= '0;
      end else begin
        if (la_v1) aq_w <= aq_w + 1'b1;
        if (aq_pop) aq_r <= aq_r + 1'b1;
        aq_n <= aq_n + 3'(la_v) - 3'(aq_pop);
      end
    always_comb begin
      act_we = (aq_pop && !ah.rec) ? ah.we : '0;
      act_row = ah.row;
      act_idx = ah.idx;
      act_data = ah.d;
      asc_we = aq_pop && ah.rec && ah.we[0];
      asc_row = ah.row;
      asc_blk = ah.idx[15:0];
      asc_data = ah.d[63:0];
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && rst_seen && la_v1 && (aq_w - aq_r) == (AQL+1)'(AQD))
        $fatal(1, "otpu_dma: LDA's ACT write queue overrun");
`endif
  end else begin : g_nla
    always_comb begin
      act_we = '0; act_row = '0; act_idx = '0; act_data = '0;
      asc_we = 1'b0; asc_row = '0; asc_blk = '0; asc_data = '0;
      aq_n = '0;
    end
  end

  // ---- ST: a two-stage read pipeline. The segment read last cycle is in t_rdata (pa: its
  // position, lanes, whether it ends its chunk), the one before in t_rq, a register (st_pend,
  // pp, pm, pl), and the chunk gathered so far in cb: the buffer's write data (st_cd) comes from
  // flip-flops, not from TMEM's block RAMs through its read path in the same cycle (the se-sys
  // build's TMEM port 1 -> lb DI paths, BRAM to BRAM, -0.38 ns). Both stages move with adv;
  // while they hold, t_rdata holds too (no new read). The segment whose request is in TMEM's
  // input stage (the request register) is pa0's (release r2), the one read pa's.
  logic                   pa0, pla0, pa, pla;
  int                     ppa0, ppa;
  logic [W-1:0]           pma0, pma;
  logic [W-1:0][31:0]     t_rq;
  logic                   st_pend, pl;
  (* max_fanout = 64 *) int pp;   // selects every data bit of b_wdata: replicated
  logic [W-1:0]           pm;
  logic [CW-1:0][31:0]    cb;
  logic [CW-1:0]          cbm;
  logic                   st_fin;                   // every chunk is in the buffer
  logic [PW:0]            st_wn;                    // chunks left in the write run
  wire st_wr = st_pend && pl;                       // the chunk is complete
  wire adv   = !st_wr || oc_nf;                     // the read -> buffer pipeline moves
  wire st_rd = busy && is_st && !ackw && sl_nz && adv;
  assign st_push = busy && is_st && !ackw && st_wr && adv;
  assign st_iss  = st_wq && b_gnt;

  // TMEM's input stage (release r2, docs/tmem_nocopy.md 7): the read request (t_*_c) goes to
  // TMEM from flip-flops, a cycle after it forms, moving with adv (a held request is asked
  // again once adv returns); the stream fill's address (fs_k's lanes) and ST's segment
  // arithmetic end at the register, not at the TMEM's ranks and pumped banks. t_rv is the
  // pending request (it moves with adv, held while adv is low); t_re_q, what TMEM reads, is
  // that request where adv will be up: adv_nx, adv a cycle ahead from its flip-flops' next
  // values, so no adv gate sits between the register and TMEM (the r2 slice OOC's worst clk ->
  // clk2x path: pl -> adv -> the gate -> the TMEM's ranks -> a pumped bank's address, 12 levels)
  logic [LANES-1:0]       t_ren_c, t_rv, t_re_q;
  logic [LANES-1:0][31:0] t_raddr_c;
  // st_pend, pl and oc_nf move in the ST branch of the process below (st_pend and pl with adv,
  // occ with the push and the issue); a start clears st_pend; elsewhere st_pend is low, so adv
  // stays up
  logic       adv_nx;
  logic [1:0] oc_st;
  always_comb begin
    oc_st = oc_nx(occ, st_push, st_iss);
    if (rst || go) adv_nx = 1'b1;
    else if (!ackw && busy && is_st)
      adv_nx = !((adv ? pa : st_pend) && (adv ? pla : pl)) || oc_st[1];
    else adv_nx = adv;
  end
  // ... and by bank (otpu_tmem PBR): t_rbv the pending request's banks (with t_rv), t_rb_q what
  // TMEM reads (with t_re_q), t_rbad their words (with t_raddr), the rotation decoded before the
  // registers, so TMEM's ranks and compares start at flip-flops (r150d, the m8 card at a real 150
  // MHz: t_re_q -> the guests' pick and the rotation -> the ranks -> a pumped bank's address, 8
  // levels, the card's worst core -> clk2x path)
  logic [LANES-1:0]       t_rben_c, t_rbv, t_rb_q;
  logic [LANES-1:0][31:0] t_rbad_c;
  otpu_tmem_breq #(.LANES(LANES)) u_tbr (.en(t_ren_c), .a0(t_raddr_c[0]), .ben(t_rben_c), .bad(t_rbad_c));
  always_ff @(posedge clk)
    if (rst) begin t_rv <= '0; t_re_q <= '0; t_rbv <= '0; t_rb_q <= '0; end
    else begin
      if (adv) begin t_rv <= t_ren_c; t_raddr <= t_raddr_c; t_rbv <= t_rben_c; t_rbad <= t_rbad_c; end
      t_re_q <= adv_nx ? (adv ? t_ren_c : t_rv) : '0;
      t_rb_q <= adv_nx ? (adv ? t_rben_c : t_rbv) : '0;
    end
  assign t_ren = t_re_q;
  assign t_rben = t_rb_q;
`ifndef SYNTHESIS
  // the prediction is adv, and TMEM reads the pending request exactly while adv is up
  logic adv_q;
  bit   adv_rst;
  initial adv_rst = 1'b0;
  always @(posedge clk) begin
    adv_q <= adv_nx;
    if (rst) adv_rst <= 1'b1;
    if (!rst && adv_rst && (adv_q != adv || t_re_q != (adv ? t_rv : '0)))
      $fatal(1, "otpu_dma: adv %0d predicted %0d, read %b of pending %b", adv, adv_q, t_re_q, t_rv);
  end
`endif
  always_comb begin
    b_req = ld_req; b_we = 1'b0; b_addr = ic;
    t_ren_c = '0; t_raddr_c = '0;
    for (int p = 0; p < SPC; p++)
      for (int l = 0; l < W; l++) begin
        st_cm[p * W + l] = cbm[p * W + l] || (pp == p && pm[l]);
        st_cd[32 * (p * W + l) +: 32] = (pp == p) ? t_rq[l] : cb[p * W + l];
      end
    b_wmask = lm[rp];
    b_wdata = lb_q;
    if (ds_wreq) begin                    // a stream: the head chunk of the gather
      b_req = 1'b1;
      b_we = 1'b1;
      b_addr = ds_wa;
      for (int p = 0; p < SPC; p++)
        for (int l = 0; l < W; l++) begin
          b_wmask[p * W + l] = 1'b1;
          b_wdata[32 * (p * W + l) +: 32] = gq[p][32 * l +: 32];
        end
    end
    if (ds_fill)
      for (int l = 0; l < W; l++) begin
        t_ren_c[l] = fk_m[l];
        t_raddr_c[l] = fk_a + 32'(l);
      end
    if (ds_dsc)                           // a STREAM's descriptor
      for (int l = 0; l < W; l++) begin
        t_ren_c[l] = 1'b1;
        t_raddr_c[l] = 32'(sc.w1[15:0]) + 32'(l);
      end
    if (busy && is_st && !ackw) begin
      b_addr = ic;
      if (st_wq) begin
        b_req = 1'b1;
        b_we = 1'b1;
      end
      if (st_rd)
        for (int l = 0; l < W; l++) begin
          t_ren_c[l] = sm[l];
          t_raddr_c[l] = so + 32'(l);
        end
    end
  end

  always_ff @(posedge clk) begin
    fr_k0 <= rst ? SF_NONE : fk_k;
    fr_i0 <= fk_i;
    fr_k <= rst ? SF_NONE : fr_k0;
    fr_i <= fr_i0;
    pe <= !rst && (ds_take || ds_drain);
    pe_in <= ds_take;
    pe_zero <= ds_zero || ds_spad;
    pe_pos <= ds_pos;
  end
  assign ds_ow = ds_out;
  for (genvar l = 0; l < W; l++) begin : g_ob
    (* ram_style = "distributed" *) logic [31:0] ob [CBD];
    always_ff @(posedge clk) if (ow_e[l]) ob[ow_a[l]] <= o_q[l % SR];
    assign ob_q[l] = ob[oi[OBW-1:0]];
  end
  always_ff @(posedge clk) o_q <= o_d;
  // LDA's fixed fields from gc / gd (pcmd's while the slot holds one): the requests' (lq_: an LDA
  // started or launched as the front) and the delivery's (la_: started, or the front handed over).
  // Loaded whatever the command (an LDA's are the only ones read), so each is an enable, no mux.
  always_ff @(posedge clk) begin
    if (LDA && (go || go_f)) begin
      lq_rl <= gc.w2[7:0] - 8'd1;
      lq_rs <= WA'(gc.w3 >> 2);
      lq_nw <= 16'(gd.n);
    end
    if (LDA && (go || hand)) begin
      la_rl <= gc.w2[7:0] - 8'd1;
      la_ab <= gc.w2[15:8];
      la_ar <= gc.w4[7:0];
      la_nw <= 16'(gd.n);
      la_ns <= 16'((gd.n + (W - 1)) >> SWL);
      la_nbs <= 16'(32'(gc.w2[31:16]) * SPC);
      la_rs <= WA'(gc.w3 >> 2);
    end
  end
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && rst_seen && ds_out && ow_e != '0) $fatal(1, "otpu_dma: ob read before its last o landed");
`endif

  // LDQ 2: the back is an LD (bk_ld), an LDA (bk_la) or an ST in its wr_idle wait (bk_sw); the
  // queued LD or LDA launches behind an LD or LDA whose requests have all gone out (not
  // completing), the LD behind an ST whose words it does not read; the back completes (bk_fin: an
  // LD's last write registered, an LDA's ACT queue drained, la_dn; an ST's writes acknowledged) and
  // hands over to the front (an ST's: also to an LD launching that cycle)
  wire bk_ld  = busy && !ackw && !is_st && !is_ds && !is_ww && !(LDA && is_la);
  wire bk_la  = LDA && busy && is_la;
  wire la_dn  = la_dr && aq_n == 0;
  wire bk_sw  = busy && ackw && is_st;
  wire bk_fin = (bk_ld && ld_last) || (bk_sw && wr_idle) || (bk_la && la_dn);
  assign go_f = LDQ > 1 && sl_q && q_ld && ((bk_ld && !cl_nz && !ld_last) || (bk_sw && !q_ov) ||
                                             (bk_la && !cl_nz && !la_dn));
  assign hand = LDQ > 1 && bk_fin && (sl_f || go_f);
  // an ST behind an ST in its wait (the header): the back's done pending in sb, done at wr_idle
  // (sb_fin; the cycle it launches, too)
  assign go_s = LDQ > 1 && sl_q && q_st && bk_sw;
  wire   sb_fin = (sb || go_s) && wr_idle;
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && rst_seen && (sl_q && sl_f || sl_f && !(busy && !is_ds && !is_ww && (!is_st || ackw)) ||
                             (sl_q || sl_f) && start))
      $fatal(1, "otpu_dma: slot queued %0d front %0d, busy %0d st %0d ackw %0d, start %0d", sl_q,
             sl_f, busy, is_st, ackw, start);
  // while the back's done is pending the DMA runs the front ST, which writes nothing until it is
  // done (so never reaches its own wait) and holds the only command: no slot, no start
  always_ff @(posedge clk)
    if (!rst && rst_seen && sb && (!busy || !is_st || ackw || st_wq || sl_q || sl_f || start))
      $fatal(1, "otpu_dma: the back ST's done pending beside busy %0d st %0d ackw %0d wq %0d slot %0d%0d start %0d",
             busy, is_st, ackw, st_wq, sl_q, sl_f, start);
`endif
  always_ff @(posedge clk) begin
    done <= ld_fin || la_fin;            // an LD is done once its last TMEM write has landed
    la_fin <= 1'b0;                      // (LDA: its queue's last entry two cycles before)
    ow_e <= '0;                          // set below for the o value taken this cycle
    ld_fin <= 1'b0;
    ld_last <= 1'b0;
    dv_v <= ld_dv;
    dv_m <= sm;
    dv_a <= so;
    dv_p <= pos_of(sw);
    if (rst) begin
      ld_fin <= 1'b0;
      dv_v <= 1'b0;
      busy <= 1'b0;
      st_pend <= 1'b0; pa <= 1'b0; pa0 <= 1'b0;
      st_wq <= 1'b0;
      ackw <= 1'b0;
      ds_wr <= 1'b0;
      ds_rr <= 1'b0;
      ds_hr <= 1'b0; ds_hw <= 1'b0;
      ds_dsc <= 1'b0; ds_dsd <= 1'b0; ds_su <= 1'b0; ds_wait <= 1'b0;
      ds_fill <= 1'b0; ds_run <= 1'b0; ds_out <= 1'b0;
      ds_fx <= 1'b0; ds_dw <= 1'b0;
      ss_req <= 1'b0;
      is_ww <= 1'b0; ww_c1 <= 1'b0; ww_c2 <= 1'b0; ww_sl <= 1'b0; ww_late <= 1'b0; err <= 1'b0;
      is_la <= 1'b0; la_dr <= 1'b0; rq_la <= 1'b0;
    end else if (go && (gc.op == OP_DSTEP || gc.op == OP_STREAM)) begin
      is_la <= 1'b0; rq_la <= 1'b0;
      // the reads start at the setup: a DSTEP's next cycle, a STREAM's once its descriptor is in
      sc <= gc;
      is_ds <= 1'b1; is_st <= 1'b0; is_ww <= 1'b0; ds_zero <= gc.flags[DF_ZERO];
      ds_nowr <= ds_sim && gc.op == OP_DSTEP && gc.flags[7];
      ds_hr <= H16 && (gc.flags[(gc.op == OP_DSTEP) ? DF_ST16 : STF_ST16] ||
                       ST16S && gc.op != OP_DSTEP && gc.flags[STF_ST16R]);
      ds_hw <= H16 && (gc.flags[(gc.op == OP_DSTEP) ? DF_ST16 : STF_ST16] ||
                       ST16S && gc.op != OP_DSTEP && gc.flags[STF_ST16W]);
      sleft <= '0; sl_nz <= 1'b0; sl_one <= 1'b0;   // no LD delivery into TMEM
      cleft <= '0; cl_nz <= 1'b0; cl_one <= 1'b0;
      occ <= '0; {oc_nf, oc_lr} <= oc_of('0); cnt <= '0; wp <= '0; rp <= '0;
      ds_ycnt <= '0; ds_ocnt <= '0; ds_wch <= '0;
      ds_dsc <= gc.op == OP_STREAM; ds_dsd <= 1'b0; ds_su <= gc.op == OP_DSTEP;
      ds_wait <= 1'b0; ds_fill <= 1'b0; ds_run <= 1'b0; ds_out <= 1'b0;
      ds_fx <= 1'b0; ds_dw <= 1'b0;
      og <= '0; gt <= '0; gh <= '0; ds_pos <= '0; oi <= '0; ds_wr <= 1'b0;
      ds_rr <= 1'b0; ds_rc <= '0;
      st_pend <= 1'b0; pa <= 1'b0; pa0 <= 1'b0;
      ackw <= 1'b0;
      busy <= 1'b1;
    end else if (go) begin
      is_st <= (gc.op == OP_ST);
      is_ds <= 1'b0;
      is_ww <= (gc.op == OP_WAITW);
      is_la <= LDA && (gc.op == OP_LDA);
      rq_la <= LDA && (gc.op == OP_LDA);
      la_dr <= 1'b0;
      if (LDA) begin
        la_ph <= '0;
        la_r <= '0;
        lq_r <= '0;
        la_sc <= '0;
        la_rec <= 1'b0;
        la_nx <= gd.a + WA'(gc.w3 >> 2);
        lq_nx <= gd.a + WA'(gc.w3 >> 2);
      end
      ww_a <= gd.a; ww_t <= gc.w2; ww_ref <= gc.w3; ww_mask <= gc.w4; ww_iv <= gc.w5;
      ww_to <= gc.w6; ww_cmp <= gc.flags[1:0]; ww_tc <= '0;
      ww_c1 <= 1'b0; ww_c2 <= 1'b0; ww_sl <= 1'b0; ww_late <= 1'b0;
      dw <= gd.a;
      de <= gd.e;
      sw <= gd.sw;
      so <= gd.so;
      sm <= gd.sm;
      sleft <= gd.ns; sl_nz <= gd.ns != 0; sl_one <= gd.ns == 1;
      ic <= gd.ic;
      cleft <= gd.nc; cl_nz <= gd.nc != 0; cl_one <= gd.nc == 1;
      occ <= '0; {oc_nf, oc_lr} <= oc_of('0); cnt <= '0; wp <= '0; rp <= '0;
      st_pend <= 1'b0; pa <= 1'b0; pa0 <= 1'b0;
      st_wq <= 1'b0;
      st_fin <= 1'b0;
      cbm <= '0;
      ackw <= 1'b0;
      if (gd.n == 0) done <= 1'b1;
      else busy <= 1'b1;
    end else if (ackw) begin
      if (fr_act) begin
        // the front's requests and returns under the ST's wait (the buffer empty, nothing eaten)
        if (ld_iss) begin
          ic <= ic + CW;
          cleft <= cleft - 1;
          cl_nz <= !cl_one; cl_one <= cleft == 2;
        end
        occ <= occ + (PW+1)'(ld_iss);
        {oc_nf, oc_lr} <= oc_nx(occ, ld_iss, 1'b0);
        cnt <= cnt + (PW+1)'(b_rvalid);
        if (b_rvalid) wp <= wp + 1'b1;
      end
      if (wr_idle) begin
        ackw <= 1'b0;
        is_ds <= 1'b0;
        done <= 1'b1;
        if (hand) begin                   // the front's LD takes the back (busy stays)
          is_st <= 1'b0;
          dw <= gd.a; de <= gd.e; sw <= gd.sw; so <= gd.so; sm <= gd.sm;
          sleft <= gd.ns; sl_nz <= gd.ns != 0; sl_one <= gd.ns == 1;
          st_pend <= 1'b0; pa <= 1'b0; pa0 <= 1'b0; st_wq <= 1'b0; st_fin <= 1'b0; cbm <= '0;
        end else
          busy <= 1'b0;
      end
    end else if (busy) begin
      if (is_ds) begin
        // a STREAM's descriptor: read, then captured (dsw)
        if (ds_dsc) begin
          ds_dsc <= 1'b0;
          ds_dw <= 1'b1;
        end
        if (ds_dw) begin
          ds_dw <= 1'b0;
          ds_dsd <= 1'b1;
        end
        if (ds_dsd) begin
          ds_dsd <= 1'b0;
          ds_su <= 1'b1;
        end
        // the setup: the reads start, SE is requested (ss_cfg valid while ss_req)
        if (ds_su) begin
          ds_su <= 1'b0;
          ic <= su_src >> 2;
          // cleft = rows * ns / SPC chunks; its flags from the factors, not the product (the
          // multiply's output feeds cleft alone)
          // (a 16-bit source: half the chunks)
          cleft <= ds_zero ? '0 : (32'(su_cfg.rows) * su_ns) >> (CWL - SWL + 32'(ds_hr));
          cl_nz <= !ds_zero && !prod_lt(su_cfg.rows, su_ns, ds_hr ? 2 * SPC : SPC);
          cl_one <= !ds_zero && !prod_lt(su_cfg.rows, su_ns, ds_hr ? 2 * SPC : SPC) &&
                    prod_lt(su_cfg.rows, su_ns, ds_hr ? 4 * SPC : 2 * SPC);
          ds_wb <= su_dst >> 2;
          ds_wa <= su_dst >> 2;
          ds_oa <= su_out;
          ds_rows <= su_cfg.rows; ds_qen <= su_cfg.q_en; ds_pad <= su_cfg.pad64;
          ds_sj <= '0; ds_yj <= '0;
          ds_nseg <= su_nseg; ds_left <= su_nseg;
          ds_nr <= 9'((32'(su_cfg.rows) + SR - 1) >> SRW);
          ds_nv <= 8'((32'(su_cfg.rows) + W - 1) / W);
          ss_cfg <= su_cfg;
          ss_req <= !ds_bes;
          ds_wait <= 1'b1;
        end
        // the fill once SE is granted: one TMEM read per cycle, captured by SE a cycle later
        if (ds_wait && (ss_gnt || ds_bes)) begin
          ds_wait <= 1'b0;
          ds_fill <= 1'b1;
        end
        if (ds_fill && fs_k == SF_NONE) begin
          ds_fill <= 1'b0;
          ds_fx <= 1'b1;
        end
        if (ds_fx) begin
          ds_fx <= 1'b0;
          ds_run <= 1'b1;
        end
        if (ds_take) begin
          ds_left <= ds_left - 1'b1;
          if (!ds_zero && !ds_spad)
            ds_pos <= (SRX ? ds_pend : ds_eat) ? '0 : ds_pos + 1'b1;
          ds_sj <= ds_sj + 1'b1;                   // pad64 rows are 16 segments
        end
        // the updated segments into the gather, a chunk out per write
        if (y_v) begin
          ds_ycnt <= ds_ycnt + 1'b1;
          ds_yj <= ds_yj + 1'b1;
        end
        if (y_land) gt <= gt + 1'b1;
        if (ds_wreq && b_gnt) begin
          gh <= gh + 1'b1;
          ds_wch <= ds_wch + 1'b1;
          ds_wa <= SRX ? sr_wan : ds_wa + CW;
        end
        og <= og_nx;
        ds_wr <= ds_wr_nx;
        // a read run: starts with a request when RUN slots are free, ends after RUN chunks
        if (ld_iss) ds_rc <= (32'(ds_rc) == RUN - 1) ? '0 : ds_rc + 1'b1;
        ds_rr <= ds_rr_nx || (ld_iss && !ds_rr && RUN > 1 && !cl_one);
        if (o_v) begin
          // (SR > 1: the row step's rows SR ocnt + g, those in range; consecutive lanes)
          for (int g = 0; g < SR; g++)
            if (!SRX || (32'(ds_ocnt) << SRW) + 32'(g) < 32'(ds_rows))
              ow_e[((32'(ds_ocnt) << SRW) + 32'(g)) % W] <= 1'b1;
          for (int l = 0; l < W; l++) ow_a[l] <= OBW'((32'(ds_ocnt) << SRW) / W);
          ds_ocnt <= ds_ocnt + 1'b1;
        end
        // SE is done once every updated segment and o is out (pe's last cycle is this one at
        // the latest, and SE sees it while ss_gnt holds)
        if (ds_run && ds_flushed) ss_req <= 1'b0;
        if (ds_run && ds_flushed && og == 0 && sr_idle) begin
          ds_run <= 1'b0;
          if (ds_qen) ds_out <= 1'b1;
          else ackw <= 1'b1;
        end
        if (ds_out) begin
          oi <= oi + 1'b1;
          if (oi + 1'b1 == ds_nv) begin
            ds_out <= 1'b0;
            ackw <= 1'b1;
          end
        end
      end
      if (ld_dv || st_rd) begin          // the TMEM side moves to the next segment
        sw <= sw + W;
        so <= so + W;
        sleft <= sleft - 1;
        sl_nz <= !sl_one; sl_one <= sleft == 2;
        for (int l = 0; l < W; l++) sm[l] <= in_rng(sw + W + 32'(l));
      end
      if (!is_st) begin
        if (ld_iss) begin
          ic <= (SRX && is_ds) ? sr_icn : ic + CW;
          cleft <= cleft - 1;
          cl_nz <= !cl_one; cl_one <= cleft == 2;
          if (LDA && rq_la && cl_one && lq_r != lq_rl) begin   // LDA: the next row's chunks
            ic <= lq_nx & ~WA'(CW - 1);
            cleft <= lq_nc; cl_nz <= 1'b1; cl_one <= lq_nc == 1;
            lq_nx <= lq_nx + lq_rs;
            lq_r <= lq_r + 1'b1;
          end
        end
        occ <= occ + (PW+1)'(ld_iss) - (PW+1)'(ld_eat);
        {oc_nf, oc_lr} <= oc_nx(occ, ld_iss, ld_eat);
        cnt <= cnt + (PW+1)'(b_rvalid) - (PW+1)'(ld_eat);
        if (b_rvalid) wp <= wp + 1'b1;
        if (ld_eat) rp <= rp + 1'b1;
        if (ld_dv && sl_one && !(LDA && is_la && la_r != la_rl)) ld_last <= 1'b1;
        if (ld_last && !is_ww) begin
          if (LDA && is_la) la_dr <= 1'b1;          // LDA: once its queue has drained
          else begin
            ld_fin <= 1'b1;
            if (hand) begin               // the front's LD takes the back (busy stays)
              dw <= gd.a; de <= gd.e; sw <= gd.sw; so <= gd.so; sm <= gd.sm;
              sleft <= gd.ns; sl_nz <= gd.ns != 0; sl_one <= gd.ns == 1;
            end else
              busy <= 1'b0;
          end
        end
        if (LDA && is_la) begin
          if (la_v) la_ph <= la_end ? '0 : la_ph + 1'b1;
          if (ld_dv) begin
            la_sc <= la_sc + 1'b1;
            if (la_sc + 1'b1 == la_nbs) la_rec <= 1'b1;
          end
          if (ld_dv && sl_one && la_r != la_rl) begin
            // the row's last segment: the next row's delivery (W aligned: docs/isa.md LDA)
            dw <= la_nx;
            de <= la_nx + WA'(la_nw);
            sw <= la_nx;
            for (int l = 0; l < W; l++) sm[l] <= 32'(l) < 32'(la_nw);
            sleft <= 32'(la_ns); sl_nz <= 1'b1; sl_one <= la_ns == 1;
            la_nx <= la_nx + la_rs;
            la_r <= la_r + 1'b1;
            la_sc <= '0;
            la_rec <= 1'b0;
          end
          if (la_dn) begin
            la_dr <= 1'b0;
            la_fin <= 1'b1;
            if (hand) begin               // the front takes the back (busy stays)
              dw <= gd.a; de <= gd.e; sw <= gd.sw; so <= gd.so; sm <= gd.sm;
              sleft <= gd.ns; sl_nz <= gd.ns != 0; sl_one <= gd.ns == 1;
            end else
              busy <= 1'b0;
          end
        end
        if (is_ww) begin
          // the timeout (ww_late) is taken with no read in flight (ww_sl: the last compare failed);
          // err stays, and the DMA busy: the slice stops (otpu_slice to_err)
          ww_tc <= ww_tc + 1;
          if (ww_to != 0 && ww_tc == ww_to) ww_late <= 1'b1;
          if (ww_sl && ww_late) err <= 1'b1;
          ww_c1 <= dv_v;
          if (dv_v) ww_w <= lb_q[32 * (ww_a % CW) +: 32];
          ww_c2 <= ww_c1;
          if (ww_c1) ww_ok <= ww_holds(ww_w & ww_mask, ww_ref, ww_cmp);
          if (ww_c2) begin
            if (ww_ok) begin
              busy <= 1'b0;
              ld_fin <= 1'b1;
            end else begin
              ww_sl <= 1'b1;
              ww_n <= ww_iv;
            end
          end
          if (ww_sl && !ww_late) begin
            if (ww_n != 0) ww_n <= ww_n - 1;
            else begin                            // read the word again: the LD of one word
              ww_sl <= 1'b0;
              sw <= ww_a & ~WA'(W - 1);
              so <= ww_t - ww_a % W;
              for (int l = 0; l < W; l++) sm[l] <= 32'(l) == ww_a % W;
              sleft <= 1; sl_nz <= 1'b1; sl_one <= 1'b1;
              ic <= ww_a & ~WA'(CW - 1);
              cleft <= 1; cl_nz <= 1'b1; cl_one <= 1'b1;
            end
          end
        end
      end else begin
        // the buffer: complete chunks in, write runs out (chunk addresses from ic, in order).
        // A run starts once SRUN chunks are in (or all of them are), counted a cycle late so
        // that its chunks' buffer writes have landed before the reads that follow them
        occ <= occ + (PW+1)'(st_push) - (PW+1)'(st_iss);
        {oc_nf, oc_lr} <= oc_nx(occ, st_push, st_iss);
        if (st_push) wp <= wp + 1'b1;
        if (st_iss) begin
          rp <= rp + 1'b1;
          ic <= ic + CW;
          st_wn <= st_wn - 1'b1;
          if (st_wn == 1) st_wq <= 1'b0;
        end
        // (an ST behind an ST: its runs once the back's done, sb_fin, is in)
        if (!st_wq && (!sb || sb_fin) && (occ >= (PW+1)'(SRUN) || (st_fin && occ != 0))) begin
          st_wq <= 1'b1;
          st_wn <= occ;
        end
        if (st_fin && !st_wq && occ == 0) ackw <= 1'b1;
        if (adv) begin
          if (st_pend) begin
            if (pl) cbm <= '0;                  // the chunk goes into the buffer this cycle
            else
              for (int l = 0; l < W; l++) begin
                cb[pp * W + l] <= t_rq[l];
                cbm[pp * W + l] <= pm[l];
              end
          end
          st_pend <= pa;
          pp <= ppa;
          pm <= pma;
          pl <= pla;
          t_rq <= t_rdata;
          pa0 <= st_rd;
          ppa0 <= pos_of(sw);
          pma0 <= sm;
          pla0 <= seg_end;
          pa <= pa0;
          ppa <= ppa0;
          pma <= pma0;
          pla <= pla0;
          if (st_pend && !pa && !pa0 && !sl_nz) begin     // the last chunk goes into the buffer
            st_pend <= 1'b0;
            st_fin <= 1'b1;
          end
        end
      end
    end
    // LDQ 2: the slot (a launch loads the front: no request goes out that cycle, see go_f)
    if (LDQ > 1) begin
      if (rst) begin
        sl_q <= 1'b0;
        sl_f <= 1'b0;
        sb <= 1'b0;
      end else begin
        if (acc) begin
          pcmd <= cmd;
          sl_q <= 1'b1;
          q_ld <= (cmd.op == OP_LD && cmd.w3 != 0) || (LDA && cmd.op == OP_LDA && cmd.w2[7:0] != 0);
          q_ov <= (gd.a < de && gd.e > dw) || (LDA && cmd.op == OP_LDA);   // (an LDA: not behind an ST)
          q_st <= cmd.op == OP_ST && cmd.w3 != 0;
        end
        // the back ST's done (go_s took the machine, so nothing else is done this cycle)
        sb <= (sb || go_s) && !sb_fin;
        if (sb_fin) done <= 1'b1;
        if (go_q || go_s) sl_q <= 1'b0;
        if (go_f) begin
          sl_q <= 1'b0;
          sl_f <= 1'b1;
          ic <= gd.ic;
          cleft <= gd.nc; cl_nz <= gd.nc != 0; cl_one <= gd.nc == 1;
          if (LDA) begin                  // (an LDA's rows: the front's requests)
            rq_la <= gc.op == OP_LDA;
            lq_r <= '0;
            lq_nx <= gd.a + WA'(gc.w3 >> 2);
          end
        end
        if (hand) begin
          sl_f <= 1'b0;
          if (LDA) begin                  // the front's delivery (an LDA: from its row 0)
            is_la <= gc.op == OP_LDA;
            la_dr <= 1'b0;
            la_ph <= '0;
            la_r <= '0;
            la_sc <= '0;
            la_rec <= 1'b0;
            la_nx <= gd.a + WA'(gc.w3 >> 2);
          end
        end
      end
    end else begin
      sl_q <= 1'b0;
      sl_f <= 1'b0;
      sb <= 1'b0;
    end
  end
endmodule
