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
// The DRAM port may refuse a request (b_gnt low) and read data may take any time to return (in
// order). An ST or a DSTEP completes once the DRAM has acknowledged all its writes (wr_idle).
module otpu_dma
  import otpu_pkg::*;
#(
  parameter int D     = 32,
  parameter int LANES = 8,
  parameter int DEPTH = 128,                          // LD chunk buffer (a power of two)
  parameter bit HAS_DSTEP = 1'b1                      // streams: DSTEP, STREAM (SE's tail in u_vpu)
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    start,
  input  cmd_t                    cmd,
  output logic                    rdy,
  output logic                    done,
  // DRAM port B (the DMA has priority on it; read responses are routed back by tag)
  output logic                    b_req,
  input  logic                    b_gnt,      // the request is taken this cycle
  output logic                    b_we,
  output logic [D/4-1:0]          b_wmask,
  output logic [D*8-1:0]          b_wdata,
  output logic [31:0]             b_addr,
  input  logic                    b_rvalid,
  input  logic [D*8-1:0]          b_rdata,
  input  logic                    wr_idle,    // no DRAM write outstanding
  // TMEM lanes (read port A, write port)
  output logic [LANES-1:0]        t_ren,
  output logic [LANES-1:0][31:0]  t_raddr,
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
  output logic [31:0]             ss_in_d [LANES],
  output logic [2:0]              ss_fk,
  output logic [4:0]              ss_fi,
  output logic [31:0]             ss_fd [LANES],
  input  logic                    ss_y_v,
  input  logic [31:0]             ss_y_d [LANES],
  input  logic                    ss_o_v,
  input  logic [31:0]             ss_o_d
);
  localparam int CW  = D / 4;                         // words per chunk
  localparam int W   = (CW < LANES) ? CW : LANES;     // words per segment
  localparam int SPC = CW / W;                        // segments per chunk
  localparam int SWL = $clog2(W);
  localparam int CWL = $clog2(CW);
  localparam int PW  = $clog2(DEPTH);
  initial if (DEPTH != (1 << PW) || DEPTH < 2) $fatal(1, "otpu_dma: DEPTH must be a power of two");

  logic        busy, is_st, ackw;
  assign rdy = !busy;
  // DSTEP: the phases (fill: TMEM reads of the vectors; run: stream; out: o to TMEM)
  logic        is_ds, ds_fill, ds_run, ds_out, ds_zero;
  logic [31:0] dw, de;                 // the DRAM word range [dw, de)
  // TMEM side: the segment at DRAM word address sw, its TMEM address, its lanes in range
  // (registered: the TMEM arbiter sees these lanes), and the segments left
  logic [31:0]  sw, so, sleft;
  logic [W-1:0] sm;

  function automatic logic in_rng(input logic [31:0] a);
    return (a >= dw) && (a < de);
  endfunction
  function automatic int pos_of(input logic [31:0] a);     // segment index within its chunk
    return int'((a % CW) / W);
  endfunction
  wire seg_end = (pos_of(sw) == SPC - 1) || (sleft == 1);   // the segment at sw ends its chunk

  // ---- LD: chunk requests (address ic, cleft left), the chunk buffer, segment delivery
  logic [31:0]   ic, cleft;
  // cleft != 0 and cleft == 1, registered with cleft: no 32-bit compare on b_req's path into
  // otpu_native_dram's queue (the build's DMA -> memory path)
  logic          cl_nz, cl_one;
  // r * n < lim (lim <= 2 * SPC), without an r x n multiply: both factors are below lim then,
  // so their product on clog2(2 * SPC) bits each decides
  localparam int PLM = (1 << $clog2(2 * SPC)) - 1;
  function automatic logic prod_lt(input logic [8:0] r, input logic [5:0] n, input int lim);
    if (r == 0 || n == 0) return 1'b1;
    if (32'(r) >= lim || 32'(n) >= lim) return 1'b0;
    return (32'(r) & PLM) * (32'(n) & PLM) < lim;
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
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && busy && (cl_nz != (cleft != 0) || cl_one != (cleft == 1)))
      $fatal(1, "otpu_dma: cl_nz %0d cl_one %0d but cleft %0d", cl_nz, cl_one, cleft);
`endif
  wire ld_req = ld_act && cl_nz && (occ != (PW+1)'(DEPTH)) &&
                (!is_ds || ds_rr || (!ds_wreq && occ <= (PW+1)'(DEPTH - RUN)));
  wire ld_iss = ld_req && b_gnt && !ds_wreq;
  wire ld_dv  = ld_act && (sleft != 0) && (cnt != 0);       // deliver the segment at sw
  wire ld_eat = (ld_dv && seg_end) || ds_eat;               // ... which frees its chunk's slot

  // block RAM, written in its own reset-free process (see otpu_mxu); read every cycle
  (* ram_style = "block" *) logic [D*8-1:0] lb [DEPTH];
  logic [D*8-1:0] lb_q;
  // ST: a complete chunk (st_push: its data st_cd, word mask st_cm) into the buffer at wp;
  // a write run (st_wq) issues the chunk at rp, and the buffer's read follows the next one
  logic           st_push, st_iss, st_wq;
  logic [D*8-1:0] st_cd;
  logic [CW-1:0]  st_cm;
  (* ram_style = "distributed" *) logic [CW-1:0] lm [DEPTH];
  always_ff @(posedge clk) begin
    if (b_rvalid) lb[wp] <= b_rdata;
    else if (st_push) lb[wp] <= st_cd;
    lb_q <= lb[st_iss ? rp + 1'b1 : rp];
  end
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
  logic [31:0]  ds_oa, ds_wb;                        // o (TMEM), the stream's write (DRAM words)
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
  // before the run: a STREAM's descriptor read (ds_dsc) and capture (ds_dsd), the setup (ds_su;
  // a DSTEP's from its command), SE's grant (ds_wait), the fill (ds_fill)
  cmd_t         sc;                                  // the DSTEP / STREAM
  logic         ds_dsc, ds_dsd, ds_su, ds_wait;
  logic [31:0]  dsw [8];                             // the STREAM's descriptor words
  logic [2:0]   fr_k;                                // the fill step read last cycle (SF_*)
  logic [4:0]   fr_i;
  logic [$clog2(SPC > 1 ? SPC : 2)-1:0] ds_pos, pe_pos;   // segment within the chunk at rp
  (* max_fanout = 64 *) logic pe;
  logic         pe_in, pe_zero;
  logic [GW-1:0] og;                                 // gathered updated segments (<= NG)
  logic [$clog2(NG)-1:0] gt;                         // next gather slot
  (* max_fanout = 64 *) logic [$clog2(NGC)-1:0] gh;  // the chunk to write next
  logic [W*32-1:0] y_p, gq [SPC];                   // gq[p]: segment p of chunk gh
  logic         y_v, o_v;
  wire          y_keep = y_v && !(ds_pad && ds_yj[3]);   // a Y segment kept (not pad64's)
  logic [31:0]  y_d [W], in_d [W], o_d;
  logic [31:0]  ob [W][CBD];
  // an o value lands in ob a cycle after it leaves the datapath: its one-hot entry enable and
  // the value, registered (o_d went to all 256 entries, their enables decoded from ds_ocnt:
  // 8 ns of wire at MCOLS=4). ds_out, the first read of ob, starts on the edge that writes
  // the last o at the earliest
  logic [W-1:0][CBD-1:0] ow_e;
  (* max_fanout = 32 *) logic [31:0] o_q;
  logic [7:0]   oi;                                  // o segment written to TMEM next
  // SE runs up to SE_LAG cycles behind pe (otpu_vpu registers the ss_* inputs) and qualifies
  // y_v and o_v with its own pe: taking one more segment needs room for it, this cycle's y and
  // the segments of the last SE_LAG pe cycles (pe's counted, the older ones as always taken)
  localparam int SE_LAG = 2;
  wire  ds_room = ({1'b0, og} + (GW+1)'(y_v) + (GW+1)'(pe)) <= (GW+1)'(NG - SE_LAG);
  wire  ds_take = ds_run && (ds_left != 0) && (ds_zero || ds_spad || cnt != 0) && ds_room;
  wire  ds_flushed = (ds_ycnt == ds_nseg) && (!ds_qen || ds_ocnt == ds_rows);
  wire  ds_drain = ds_run && (ds_left == 0) && !ds_flushed && ds_room;
  assign ds_eat = ds_take && !ds_zero && !ds_spad && (32'(ds_pos) == SPC - 1);
  // a registered request (it selects the 1024-bit write data: replicated, no decode on the
  // path); it follows og, which only counts during a stream
  (* max_fanout = 64 *) logic ds_wr;
  wire  [GW-1:0] og_nx = og + GW'(y_keep) - ((ds_wreq && b_gnt) ? GW'(SPC) : GW'(0));
  wire  ds_rr_nx = ds_rr && !(ld_iss && (32'(ds_rc) == RUN - 1 || cl_one));
  // a write run: once RUN chunks are in (or the stream has ended), then while chunks are in
  wire  ds_wr_nx = (og_nx >= GW'(SPC)) &&
                   (ds_wr || (!ds_rr_nx && (og_nx >= GW'(RUN * SPC) || ds_left == 0)));
  assign ds_wreq = ds_wr;
  // the gather: one simple dual-port RAM per segment position (write: the segment from SE at
  // chunk gt / SPC; read: chunk gh)
  for (genvar p = 0; p < SPC; p++) begin : g_gb
    (* ram_style = "distributed" *) logic [W*32-1:0] gb [NGC];
    always_ff @(posedge clk)
      if (y_keep && 32'(gt) % SPC == p) gb[32'(gt) / SPC] <= y_p;
    assign gq[p] = gb[gh];
  end
  always_comb
    for (int l = 0; l < W; l++) begin
      y_p[32 * l +: 32] = y_d[l];
      in_d[l] = pe_zero ? 32'd0 : lb_q[32 * (32'(pe_pos) * W + l) +: 32];
    end

  // SE's ports (the tail and the reduction live in otpu_vpu)
  assign ss_pe = pe;
  assign ss_in_v = pe_in;
  assign ss_fk = fr_k;
  assign ss_fi = fr_i;
  for (genvar l = 0; l < LANES; l++) begin : g_ss
    assign ss_in_d[l] = (l < W) ? in_d[l % W] : 32'd0;
    assign ss_fd[l] = t_rdata[l];
  end
  if (HAS_SS) begin : g_ss_on
    assign y_v = ss_y_v;
    assign o_v = ss_o_v;
    assign o_d = ss_o_d;
    for (genvar l = 0; l < W; l++) begin : g_y
      assign y_d[l] = ss_y_d[l];
    end
  end else begin : g_nods
    assign y_v = 1'b0;
    assign o_v = 1'b0;
    assign o_d = '0;
    for (genvar l = 0; l < W; l++) begin : g_y
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
  logic [31:0] su_src, su_dst, su_vec, su_x, su_k, su_ks, su_out;
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
    su_src = sc.w2; su_dst = sc.w3; su_vec = sc.w4; su_x = sc.w5; su_k = sc.w6;
    su_ks = 32'(sc.w1[31:16]); su_out = sc.w7;
    su_ok = fs_ok && dm_ok && g_ok && a_ok && rows != 0 && rows <= 12'd256 &&
            cols[5:0] == 0 && cols != 0 && cols <= 12'd256 && d4 == 0 &&
            d1[6:5] == 2'd1 && d1[13:12] == 2'd0 && d1[16:14] == 3'd1 && d1[19:17] == 3'd2 &&
            (!d1[20] || d1[22:21] == 2'd0) && !d1[23] && d2[13:4] == 0 &&
            !sc.flags[STF_SRC_T] && !sc.flags[STF_DST_T] && !sc.flags[STF_NODST];
    if (sc.op == OP_DSTEP) begin
      su_cfg.a_en = 1'b1; su_cfg.a_sel = 1'b0; su_cfg.dmode = SD_DELTA; su_cfg.g_src = SG_K0;
      su_cfg.q_en = 1'b1;
      su_src = sc.w1; su_dst = sc.w1; su_vec = sc.w2; su_x = sc.w3; su_k = sc.w5;
      su_ks = sc.w7; su_out = sc.w6;
      su_ok = 1'b1;
    end
    su_ns = su_cfg.ns;
    su_cfg.pad64 = su_ns == 6'd8;
    if (su_cfg.pad64) su_cfg.ns = 6'd16;
    su_nseg = 16'(32'(su_cfg.rows) * 32'(su_cfg.ns));
  end
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && ds_su && sc.op == OP_STREAM &&
        (!su_ok || su_src % D != 0 ||
         (su_dst != su_src && su_dst < su_src + 32'(su_cfg.rows) * su_ns * W * 4 &&
          su_src < su_dst + 32'(su_cfg.rows) * su_ns * W * 4)))
      $fatal(1, "otpu_dma: STREAM's descriptor at %0d is not in the hardware subset", sc.w1[15:0]);
`endif

  // the fill: one TMEM read of W words per cycle. Its step (fs_*: kind SF_*, index, address,
  // lanes) is registered and the next one computed a cycle ahead, so the counts stay off the
  // TMEM read address (clk125's DSTEP path at 125.49 MHz was counts -> compares -> address ->
  // TMEM bank address). The kinds go in the order q, k, a, g, x, K0, K1 (docs/stream.md 4.3),
  // skipping those the stream does not use (f_nk).
  logic [2:0]  fk_k, fs_k;
  logic [4:0]  fk_i, fs_i;
  logic [31:0] fk_a, fs_a;
  logic [W-1:0] fk_m, fs_m;
  logic [2:0]  f_nk [8];                   // the kind after each kind (SF_NONE: the fill ends)
  logic [4:0]  f_kn [8];                   // each kind's steps less one
  logic [31:0] f_ka [8];                   // each kind's first address
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
    end else if (fk_on) begin
      if (fs_i == f_kn[fs_k]) begin
        fs_k <= f_nx; fs_i <= '0; fs_a <= f_ka[f_nx]; fs_m <= f_mask(f_nx, 5'd0, ds_rows);
      end else begin
        fs_i <= fs_i + 1'b1; fs_a <= fs_a + W; fs_m <= f_mask(fs_k, fs_i + 1'b1, ds_rows);
      end
    end
    if (ds_dsd)
      for (int i = 0; i < 8; i++) dsw[i] <= t_rdata[i % LANES];
  end
  assign fk_k = fk_on ? fs_k : SF_NONE;
  assign fk_i = fk_on ? fs_i : 5'd0;
  assign fk_a = fk_on ? fs_a : 32'd0;
  assign fk_m = fk_on ? fs_m : '0;

  // the delivered segment (in lb_q) goes to the TMEM write register; its position in the
  // chunk, lanes and TMEM address come with it (dv_*)
  logic                   dv_v, ld_last;
  logic [W-1:0]           dv_m;
  logic [31:0]            dv_a;
  int                     dv_p;
  logic [LANES-1:0]       lw_en;
  logic [LANES-1:0][31:0] lw_addr, lw_data;
  logic                   ds_ow;               // DSTEP: o segment oi goes to TMEM
  always_comb begin
    lw_en = '0; lw_addr = '0; lw_data = '0;
    for (int l = 0; l < W; l++) begin
      lw_en[l] = dv_v && dv_m[l];
      lw_addr[l] = dv_a + 32'(l);
      lw_data[l] = lb_q[32 * (dv_p * W + l) +: 32];
      if (ds_ow) begin
        lw_en[l] = (32'(oi) * W + l) < 32'(ds_rows);
        lw_addr[l] = ds_oa + 32'(oi) * W + 32'(l);
        lw_data[l] = ob[l][oi[4:0]];
      end
    end
  end
  always_ff @(posedge clk) begin
    t_wen <= rst ? '0 : lw_en;
    t_waddr <= lw_addr;
    t_wdata <= lw_data;
  end

  // ---- ST: a two-stage read pipeline. The segment read last cycle is in t_rdata (pa: its
  // position, lanes, whether it ends its chunk), the one before in t_rq, a register (st_pend,
  // pp, pm, pl), and the chunk gathered so far in cb: the buffer's write data (st_cd) comes from
  // flip-flops, not from TMEM's block RAMs through its read path in the same cycle (the se-sys
  // build's TMEM port 1 -> lb DI paths, BRAM to BRAM, -0.38 ns). Both stages move with adv;
  // while they hold, t_rdata holds too (no new read)
  logic                   pa, pla;
  int                     ppa;
  logic [W-1:0]           pma;
  logic [W-1:0][31:0]     t_rq;
  logic                   st_pend, pl;
  (* max_fanout = 64 *) int pp;   // selects every data bit of b_wdata: replicated
  logic [W-1:0]           pm;
  logic [CW-1:0][31:0]    cb;
  logic [CW-1:0]          cbm;
  logic                   st_fin;                   // every chunk is in the buffer
  logic [PW:0]            st_wn;                    // chunks left in the write run
  wire st_wr = st_pend && pl;                       // the chunk is complete
  wire adv   = !st_wr || (occ != (PW+1)'(DEPTH));   // the read -> buffer pipeline moves
  wire st_rd = busy && is_st && !ackw && (sleft != 0) && adv;
  assign st_push = busy && is_st && !ackw && st_wr && adv;
  assign st_iss  = st_wq && b_gnt;

  always_comb begin
    b_req = ld_req; b_we = 1'b0; b_addr = ic;
    t_ren = '0; t_raddr = '0;
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
      b_addr = ds_wb + 32'(ds_wch) * CW;
      for (int p = 0; p < SPC; p++)
        for (int l = 0; l < W; l++) begin
          b_wmask[p * W + l] = 1'b1;
          b_wdata[32 * (p * W + l) +: 32] = gq[p][32 * l +: 32];
        end
    end
    if (ds_fill)
      for (int l = 0; l < W; l++) begin
        t_ren[l] = fk_m[l];
        t_raddr[l] = fk_a + 32'(l);
      end
    if (ds_dsc)                           // a STREAM's descriptor
      for (int l = 0; l < W; l++) begin
        t_ren[l] = 1'b1;
        t_raddr[l] = 32'(sc.w1[15:0]) + 32'(l);
      end
    if (busy && is_st && !ackw) begin
      b_addr = ic;
      if (st_wq) begin
        b_req = 1'b1;
        b_we = 1'b1;
      end
      if (st_rd)
        for (int l = 0; l < W; l++) begin
          t_ren[l] = sm[l];
          t_raddr[l] = so + 32'(l);
        end
    end
  end

  always_ff @(posedge clk) begin
    fr_k <= rst ? SF_NONE : fk_k;
    fr_i <= fk_i;
    pe <= !rst && (ds_take || ds_drain);
    pe_in <= ds_take;
    pe_zero <= ds_zero || ds_spad;
    pe_pos <= ds_pos;
  end
  assign ds_ow = ds_out;
  always_ff @(posedge clk) begin
    for (int l = 0; l < W; l++)
      for (int e = 0; e < CBD; e++)
        if (ow_e[l][e]) ob[l][e] <= o_q;
    o_q <= o_d;
  end
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && ds_out && ow_e != '0) $fatal(1, "otpu_dma: ob read before its last o landed");
`endif

  logic ld_fin;                          // LD: the last write is in the write register
  always_ff @(posedge clk) begin
    done <= ld_fin;                      // an LD is done once its last TMEM write has landed
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
      st_pend <= 1'b0; pa <= 1'b0;
      st_wq <= 1'b0;
      ackw <= 1'b0;
      ds_wr <= 1'b0;
      ds_rr <= 1'b0;
      ds_dsc <= 1'b0; ds_dsd <= 1'b0; ds_su <= 1'b0; ds_wait <= 1'b0;
      ds_fill <= 1'b0; ds_run <= 1'b0; ds_out <= 1'b0;
      ss_req <= 1'b0;
    end else if (start && (cmd.op == OP_DSTEP || cmd.op == OP_STREAM)) begin
      // the reads start at the setup: a DSTEP's next cycle, a STREAM's once its descriptor is in
      sc <= cmd;
      is_ds <= 1'b1; is_st <= 1'b0; ds_zero <= cmd.flags[DF_ZERO];
      sleft <= '0;                               // no LD delivery into TMEM
      cleft <= '0; cl_nz <= 1'b0; cl_one <= 1'b0;
      occ <= '0; cnt <= '0; wp <= '0; rp <= '0;
      ds_ycnt <= '0; ds_ocnt <= '0; ds_wch <= '0;
      ds_dsc <= cmd.op == OP_STREAM; ds_dsd <= 1'b0; ds_su <= cmd.op == OP_DSTEP;
      ds_wait <= 1'b0; ds_fill <= 1'b0; ds_run <= 1'b0; ds_out <= 1'b0;
      og <= '0; gt <= '0; gh <= '0; ds_pos <= '0; oi <= '0; ds_wr <= 1'b0;
      ds_rr <= 1'b0; ds_rc <= '0;
      st_pend <= 1'b0; pa <= 1'b0;
      ackw <= 1'b0;
      busy <= 1'b1;
    end else if (start) begin
      // one carry chain each: the counts from the start's offset in its segment / chunk
      logic [31:0] a, n, ow, oc, nc;
      a = cmd.w1 >> 2;
      n = cmd.w3;
      ow = a % W;
      oc = a % CW;
      is_st <= (cmd.op == OP_ST);
      is_ds <= 1'b0;
      dw <= a;
      de <= a + n;
      sw <= a & ~32'(W - 1);
      so <= cmd.w2 - ow;
      for (int l = 0; l < W; l++) sm[l] <= (32'(l) >= ow) && (n > 32'(l) - ow);
      sleft <= (n + (ow + (W - 1))) >> SWL;
      ic <= a & ~32'(CW - 1);
      nc = (n + (oc + (CW - 1))) >> CWL;
      cleft <= nc; cl_nz <= nc != 0; cl_one <= nc == 1;
      occ <= '0; cnt <= '0; wp <= '0; rp <= '0;
      st_pend <= 1'b0; pa <= 1'b0;
      st_wq <= 1'b0;
      st_fin <= 1'b0;
      cbm <= '0;
      ackw <= 1'b0;
      if (cmd.w3 == 0) done <= 1'b1;
      else busy <= 1'b1;
    end else if (ackw) begin
      if (wr_idle) begin
        ackw <= 1'b0;
        busy <= 1'b0;
        is_ds <= 1'b0;
        done <= 1'b1;
      end
    end else if (busy) begin
      if (is_ds) begin
        // a STREAM's descriptor: read, then captured (dsw)
        if (ds_dsc) begin
          ds_dsc <= 1'b0;
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
          cleft <= ds_zero ? '0 : (32'(su_cfg.rows) * su_ns) >> (CWL - SWL);
          cl_nz <= !ds_zero && !prod_lt(su_cfg.rows, su_ns, SPC);
          cl_one <= !ds_zero && !prod_lt(su_cfg.rows, su_ns, SPC) &&
                    prod_lt(su_cfg.rows, su_ns, 2 * SPC);
          ds_wb <= su_dst >> 2;
          ds_oa <= su_out;
          ds_rows <= su_cfg.rows; ds_qen <= su_cfg.q_en; ds_pad <= su_cfg.pad64;
          ds_sj <= '0; ds_yj <= '0;
          ds_nseg <= su_nseg; ds_left <= su_nseg;
          ds_nv <= 8'((32'(su_cfg.rows) + W - 1) / W);
          ss_cfg <= su_cfg;
          ss_req <= 1'b1;
          ds_wait <= 1'b1;
        end
        // the fill once SE is granted: one TMEM read per cycle, captured by SE a cycle later
        if (ds_wait && ss_gnt) begin
          ds_wait <= 1'b0;
          ds_fill <= 1'b1;
        end
        if (ds_fill && fs_k == SF_NONE) begin
          ds_fill <= 1'b0;
          ds_run <= 1'b1;
        end
        if (ds_take) begin
          ds_left <= ds_left - 1'b1;
          if (!ds_zero && !ds_spad) ds_pos <= ds_eat ? '0 : ds_pos + 1'b1;
          ds_sj <= ds_sj + 1'b1;                   // pad64 rows are 16 segments
        end
        // the updated segments into the gather, a chunk out per write
        if (y_v) begin
          ds_ycnt <= ds_ycnt + 1'b1;
          ds_yj <= ds_yj + 1'b1;
        end
        if (y_keep) gt <= gt + 1'b1;
        if (ds_wreq && b_gnt) begin
          gh <= gh + 1'b1;
          ds_wch <= ds_wch + 1'b1;
        end
        og <= og_nx;
        ds_wr <= ds_wr_nx;
        // a read run: starts with a request when RUN slots are free, ends after RUN chunks
        if (ld_iss) ds_rc <= (32'(ds_rc) == RUN - 1) ? '0 : ds_rc + 1'b1;
        ds_rr <= ds_rr_nx || (ld_iss && !ds_rr && RUN > 1 && !cl_one);
        if (o_v) begin
          ow_e[ds_ocnt % W][ds_ocnt / W] <= 1'b1;
          ds_ocnt <= ds_ocnt + 1'b1;
        end
        // SE is done once every updated segment and o is out (pe's last cycle is this one at
        // the latest, and SE sees it while ss_gnt holds)
        if (ds_run && ds_flushed) ss_req <= 1'b0;
        if (ds_run && ds_flushed && og == 0) begin
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
        for (int l = 0; l < W; l++) sm[l] <= in_rng(sw + W + 32'(l));
      end
      if (!is_st) begin
        if (ld_iss) begin
          ic <= ic + CW;
          cleft <= cleft - 1;
          cl_nz <= !cl_one; cl_one <= cleft == 2;
        end
        occ <= occ + (PW+1)'(ld_iss) - (PW+1)'(ld_eat);
        cnt <= cnt + (PW+1)'(b_rvalid) - (PW+1)'(ld_eat);
        if (b_rvalid) wp <= wp + 1'b1;
        if (ld_eat) rp <= rp + 1'b1;
        if (ld_dv && sleft == 1) ld_last <= 1'b1;
        if (ld_last) begin
          busy <= 1'b0;
          ld_fin <= 1'b1;
        end
      end else begin
        // the buffer: complete chunks in, write runs out (chunk addresses from ic, in order).
        // A run starts once SRUN chunks are in (or all of them are), counted a cycle late so
        // that its chunks' buffer writes have landed before the reads that follow them
        occ <= occ + (PW+1)'(st_push) - (PW+1)'(st_iss);
        if (st_push) wp <= wp + 1'b1;
        if (st_iss) begin
          rp <= rp + 1'b1;
          ic <= ic + CW;
          st_wn <= st_wn - 1'b1;
          if (st_wn == 1) st_wq <= 1'b0;
        end
        if (!st_wq && (occ >= (PW+1)'(SRUN) || (st_fin && occ != 0))) begin
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
          pa <= st_rd;
          ppa <= pos_of(sw);
          pma <= sm;
          pla <= seg_end;
          if (st_pend && !pa && sleft == 0) begin  // the last chunk goes into the buffer
            st_pend <= 1'b0;
            st_fin <= 1'b1;
          end
        end
      end
    end
  end
endmodule
