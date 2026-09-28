// LD / ST: move 32-bit words between the slice DRAM and TMEM using the DRAM burst port (B).
// DRAM side: every D-byte chunk the transfer touches is requested once (B requests are chunk
// aligned). TMEM side: the transfer is cut into W = min(D/4, LANES)-word segments aligned in
// DRAM, one per cycle (CW/W per chunk), so the lanes need no shifter: lane l carries word l of
// the segment, and the partial segments at the ends are masked. The segment's words sit in
// consecutive TMEM words, hence distinct banks.
//   LD  the chunks are requested ahead into a DEPTH-chunk buffer (block RAM). A chunk's slot is
//       reserved when it is requested, so read data (which cannot be refused) always has room.
//       A segment leaves the buffer through its registered read and the TMEM write register.
//   ST  the segments read from TMEM are gathered into a chunk register; a chunk is written once,
//       word-masked, with its last segment in range (that segment comes straight from TMEM).
//   DSTEP one Gated DeltaNet head step on an fp32 state in DRAM (docs/isa.md), updated in place:
//       q | k, v, the decay and beta are first read from TMEM into the datapath (otpu_dstep);
//       then the state's chunks are requested like an LD's and their segments fed to the
//       datapath, one per cycle, and the updated segments gathered into chunks and written
//       back; finally o goes to TMEM through the write register. The DRAM sees runs: the
//       state's reads go out RUN chunks at a time (one burst per channel) and the writes once
//       RUN chunks are gathered (NGC-chunk gather), so a read / write turnaround and a read
//       transaction's fixed cost are paid per run, not per chunk; a run of one kind never
//       starts inside a run of the other. The datapath advances with `pe`, a register: a
//       cycle it may take a segment (or, after the last one, a bubble) needs a segment in the
//       buffer and room for two more updated segments.
// The DRAM port may refuse a request (b_gnt low) and read data may take any time to return (in
// order). An ST or a DSTEP completes once the DRAM has acknowledged all its writes (wr_idle).
module otpu_dma
  import otpu_pkg::*;
#(
  parameter int D     = 32,
  parameter int LANES = 8,
  parameter int DEPTH = 128,                          // LD chunk buffer (a power of two)
  parameter bit HAS_DSTEP = 1'b1                      // the DSTEP datapath (0: none, ~30K LUT)
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
  output logic [LANES-1:0][31:0]  t_wdata
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
  logic [PW:0]   occ, cnt;             // chunks requested / received, and not yet delivered
  logic [PW-1:0] wp, rp;               // buffer slot of the next chunk received / delivered
  logic ds_wreq, ds_eat;                                    // DSTEP: a chunk write; a chunk used
  wire ld_act = busy && !is_st && !ackw;
  localparam int RUN = 16;                            // DSTEP: chunks per DRAM read / write run
  logic ds_rr;                                        // DSTEP: inside a read run
  logic [$clog2(RUN)-1:0] ds_rc;                      // ... its chunks issued
  wire ld_req = ld_act && (cleft != 0) && (occ != (PW+1)'(DEPTH)) &&
                (!is_ds || ds_rr || (!ds_wreq && occ <= (PW+1)'(DEPTH - RUN)));
  wire ld_iss = ld_req && b_gnt && !ds_wreq;
  wire ld_dv  = ld_act && (sleft != 0) && (cnt != 0);       // deliver the segment at sw
  wire ld_eat = (ld_dv && seg_end) || ds_eat;               // ... which frees its chunk's slot

  // block RAM, written in its own reset-free process (see otpu_mxu); read every cycle
  (* ram_style = "block" *) logic [D*8-1:0] lb [DEPTH];
  logic [D*8-1:0] lb_q;
  always_ff @(posedge clk) begin
    if (b_rvalid) lb[wp] <= b_rdata;
    lb_q <= lb[rp];
  end
  // ---- DSTEP
  localparam int NGC = 2 * RUN;                      // output gather (chunks)
  localparam int NG = NGC * SPC;                     // ... in segments
  localparam int GW = $clog2(NG) + 1;
  localparam int CBD = 256 / W;                      // column / row buffers per lane
  logic [31:0]  ds_qa, ds_va, ds_ga, ds_gs, ds_oa, ds_wb;
  logic [5:0]   ds_ns;                               // segments per state row
  logic [8:0]   ds_rows;
  logic [15:0]  ds_nseg, ds_left, ds_ycnt;           // segments: all, still to feed, updated
  logic [8:0]   ds_ocnt;                             // o values out of the datapath
  logic [15:0]  ds_wch;                              // chunks written
  // fill: step fk of fn (q segments, k segments, v segments, decay, beta), data a cycle later
  logic [7:0]   fk, fn, ds_nq, ds_nv;
  logic [2:0]   fr_k;                                // the step read last cycle: 1 q .. 5 beta
  logic [4:0]   fr_i;
  logic [$clog2(SPC > 1 ? SPC : 2)-1:0] ds_pos, pe_pos;   // segment within the chunk at rp
  (* max_fanout = 64 *) logic pe;
  logic         pe_in, pe_zero;
  logic [GW-1:0] og;                                 // gathered updated segments (<= NG)
  logic [$clog2(NG)-1:0] gt;                         // next gather slot
  (* max_fanout = 64 *) logic [$clog2(NGC)-1:0] gh;  // the chunk to write next
  logic [W*32-1:0] y_p, gq [SPC];                   // gq[p]: segment p of chunk gh
  logic         y_v, o_v;
  logic [31:0]  y_d [W], in_d [W], fdat [W], o_d;
  logic [31:0]  ob [W][CBD];
  logic [7:0]   oi;                                  // o segment written to TMEM next
  wire  ds_room = ({1'b0, og} + (GW+1)'(pe && y_v)) <= (GW+1)'(NG - 1);
  wire  ds_take = ds_run && (ds_left != 0) && (ds_zero || cnt != 0) && ds_room;
  wire  ds_flushed = (ds_ycnt == ds_nseg) && (ds_ocnt == ds_rows);
  wire  ds_drain = ds_run && (ds_left == 0) && !ds_flushed && ds_room;
  assign ds_eat = ds_take && !ds_zero && (32'(ds_pos) == SPC - 1);
  // a registered request (it selects the 1024-bit write data: replicated, no decode on the
  // path); it follows og, which only counts during a DSTEP
  (* max_fanout = 64 *) logic ds_wr;
  wire  [GW-1:0] og_nx = og + GW'(pe && y_v) - ((ds_wreq && b_gnt) ? GW'(SPC) : GW'(0));
  wire  ds_rr_nx = ds_rr && !(ld_iss && (32'(ds_rc) == RUN - 1 || cleft == 1));
  // a write run: once RUN chunks are in (or the stream has ended), then while chunks are in
  wire  ds_wr_nx = (og_nx >= GW'(SPC)) &&
                   (ds_wr || (!ds_rr_nx && (og_nx >= GW'(RUN * SPC) || ds_left == 0)));
  assign ds_wreq = ds_wr;
  // the gather: one simple dual-port RAM per segment position (write: the segment from the
  // datapath at chunk gt / SPC; read: chunk gh)
  for (genvar p = 0; p < SPC; p++) begin : g_gb
    (* ram_style = "distributed" *) logic [W*32-1:0] gb [NGC];
    always_ff @(posedge clk)
      if (pe && y_v && 32'(gt) % SPC == p) gb[32'(gt) / SPC] <= y_p;
    assign gq[p] = gb[gh];
  end
  always_comb
    for (int l = 0; l < W; l++) begin
      y_p[32 * l +: 32] = y_d[l];
      in_d[l] = pe_zero ? 32'd0 : lb_q[32 * (32'(pe_pos) * W + l) +: 32];
      fdat[l] = t_rdata[l];
    end

  // the datapath is built for 8 lanes (its isum_64 partial loop needs 64 / W > the adder's
  // latency); other widths have no DSTEP (the compiler emits the VOP sequence)
  if (W == 8 && HAS_DSTEP) begin : g_ds
    otpu_dstep #(.LANES(W)) u_ds (
      .clk, .rst, .init(start), .ns(ds_ns),
      .fq(fr_k == 3'd1), .fk(fr_k == 3'd2), .fv(fr_k == 3'd3), .fe(fr_k == 3'd4),
      .fb(fr_k == 3'd5), .fi(fr_i), .fd(fdat),
      .pe, .in_v(pe_in), .in_d, .y_v, .y_d, .o_v, .o_d);
  end else begin : g_nods
    assign y_v = 1'b0;
    assign o_v = 1'b0;
    assign o_d = '0;
    for (genvar l = 0; l < W; l++) begin : g_y
      assign y_d[l] = '0;
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk)
      if (!rst && is_ds) $fatal(1, "otpu_dma: no DSTEP here (W = %0d, HAS_DSTEP = %0d)", W, HAS_DSTEP);
`endif
  end

  // the fill's TMEM read this cycle: kind, index, address, lanes. Step fk's are registered
  // (fs_*), computed a cycle ahead from fk_nx: the compare / subtract chain on fk and the
  // counts stays off the TMEM read address (it was clk125's DSTEP path at 125.49 MHz:
  // ds_nq -> compares -> fk_a -> t_raddr -> TMEM bank address, 13 levels). A DSTEP primes
  // the first step for a cycle (ds_prime) before its fill starts.
  logic        ds_prime;
  logic [2:0]  fk_k, fs_k, fn_k;
  logic [4:0]  fk_i, fs_i, fn_i;
  logic [31:0] fk_a, fs_a, fn_a;
  logic [W-1:0] fk_m, fs_m, fn_m;
  wire  [7:0]  fk_nx = ds_prime ? 8'd0 : fk + 8'd1;
  always_comb begin
    fn_k = 3'd0; fn_i = '0; fn_a = '0; fn_m = '0;
    if (fk_nx < ds_nq) begin
      fn_k = 3'd1; fn_i = 5'(fk_nx); fn_a = ds_qa + 32'(fk_nx) * W; fn_m = '1;
    end else if (fk_nx < 2 * ds_nq) begin
      fn_k = 3'd2; fn_i = 5'(fk_nx - ds_nq); fn_a = ds_qa + 32'(fk_nx) * W; fn_m = '1;
    end else if (fk_nx < 2 * ds_nq + ds_nv) begin
      fn_k = 3'd3; fn_i = 5'(fk_nx - 2 * ds_nq); fn_a = ds_va + 32'(fn_i) * W;
      for (int l = 0; l < W; l++) fn_m[l] = (32'(fn_i) * W + l) < 32'(ds_rows);
    end else if (fk_nx == 2 * ds_nq + ds_nv) begin
      fn_k = 3'd4; fn_a = ds_ga; fn_m[0] = 1'b1;
    end else begin
      fn_k = 3'd5; fn_a = ds_ga + ds_gs; fn_m[0] = 1'b1;
    end
  end
  always_ff @(posedge clk)
    if (ds_prime || (ds_fill && fk != fn)) begin
      fs_k <= fn_k; fs_i <= fn_i; fs_a <= fn_a; fs_m <= fn_m;
    end
  wire fk_on = ds_fill && fk != fn;
  assign fk_k = fk_on ? fs_k : 3'd0;
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

  // ---- ST: the segment read last cycle, pending in t_rdata (its position, lanes, chunk, and
  // whether it ends its chunk), and the chunk gathered so far
  logic                   st_pend, pl;
  (* max_fanout = 64 *) int pp;   // selects every data bit of b_wdata: replicated
  logic [W-1:0]           pm;
  logic [31:0]            pc;
  logic [CW-1:0][31:0]    cb;
  logic [CW-1:0]          cbm;
  wire st_wr = st_pend && pl;                       // the chunk's write request
  wire adv   = !st_wr || b_gnt;                     // the read -> write pipeline moves
  wire st_rd = busy && is_st && !ackw && (sleft != 0) && adv;

  always_comb begin
    b_req = ld_req; b_we = 1'b0; b_addr = ic;
    t_ren = '0; t_raddr = '0;
    for (int p = 0; p < SPC; p++)
      for (int l = 0; l < W; l++) begin
        b_wmask[p * W + l] = cbm[p * W + l] || (pp == p && pm[l]);
        b_wdata[32 * (p * W + l) +: 32] = (pp == p) ? t_rdata[l] : cb[p * W + l];
      end
    if (ds_wreq) begin                    // DSTEP: the head chunk of the gather
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
    if (busy && is_st && !ackw) begin
      b_addr = pc;
      if (st_wr) begin
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
    fr_k <= rst ? 3'd0 : fk_k;
    fr_i <= fk_i;
    pe <= !rst && (ds_take || ds_drain);
    pe_in <= ds_take;
    pe_zero <= ds_zero;
    pe_pos <= ds_pos;
  end
  assign ds_ow = ds_out;

  logic ld_fin;                          // LD: the last write is in the write register
  always_ff @(posedge clk) begin
    done <= ld_fin;                      // an LD is done once its last TMEM write has landed
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
      st_pend <= 1'b0;
      ackw <= 1'b0;
      ds_wr <= 1'b0;
      ds_rr <= 1'b0;
      ds_prime <= 1'b0;
    end else if (start && cmd.op == OP_DSTEP) begin
      logic [31:0] a;
      logic [8:0]  rows;
      logic [8:0]  cols;
      logic [5:0]  ns;
      logic [7:0]  nv;
      a = cmd.w1 >> 2;
      rows = 9'(cmd.w4[15:0]);
      cols = 9'(cmd.w4[31:16]);
      ns = 6'(cols / W);
      nv = 8'((32'(rows) + W - 1) / W);
      is_ds <= 1'b1; is_st <= 1'b0; ds_zero <= cmd.flags[DF_ZERO];
      sleft <= '0;                               // no LD delivery into TMEM
      ic <= a;
      cleft <= cmd.flags[DF_ZERO] ? '0 : (32'(rows) * 32'(cols)) >> CWL;
      occ <= '0; cnt <= '0; wp <= '0; rp <= '0;
      ds_qa <= cmd.w2; ds_va <= cmd.w3; ds_ga <= cmd.w5; ds_oa <= cmd.w6; ds_gs <= cmd.w7;
      ds_wb <= a;
      ds_ns <= ns; ds_rows <= rows;
      ds_nseg <= 16'(32'(rows) * 32'(ns)); ds_left <= 16'(32'(rows) * 32'(ns));
      ds_ycnt <= '0; ds_ocnt <= '0; ds_wch <= '0;
      ds_nq <= 8'(ns); ds_nv <= nv; fk <= '0; fn <= 8'(2 * ns) + nv + 8'd2;
      ds_fill <= 1'b0; ds_prime <= 1'b1; ds_run <= 1'b0; ds_out <= 1'b0;
      og <= '0; gt <= '0; gh <= '0; ds_pos <= '0; oi <= '0; ds_wr <= 1'b0;
      ds_rr <= 1'b0; ds_rc <= '0;
      st_pend <= 1'b0;
      ackw <= 1'b0;
      busy <= 1'b1;
    end else if (start) begin
      // one carry chain each: the counts from the start's offset in its segment / chunk
      logic [31:0] a, n, ow, oc;
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
      cleft <= (n + (oc + (CW - 1))) >> CWL;
      occ <= '0; cnt <= '0; wp <= '0; rp <= '0;
      st_pend <= 1'b0;
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
        // fill: one TMEM read per cycle, captured by the datapath a cycle later
        if (ds_prime) begin
          ds_prime <= 1'b0;
          ds_fill <= 1'b1;
        end
        if (ds_fill) begin
          if (fk != fn) fk <= fk + 1'b1;
          else begin
            ds_fill <= 1'b0;
            ds_run <= 1'b1;
          end
        end
        if (ds_take) begin
          ds_left <= ds_left - 1'b1;
          if (!ds_zero) ds_pos <= ds_eat ? '0 : ds_pos + 1'b1;
        end
        // the updated segments into the gather, a chunk out per write
        if (pe && y_v) begin
          gt <= gt + 1'b1;
          ds_ycnt <= ds_ycnt + 1'b1;
        end
        if (ds_wreq && b_gnt) begin
          gh <= gh + 1'b1;
          ds_wch <= ds_wch + 1'b1;
        end
        og <= og_nx;
        ds_wr <= ds_wr_nx;
        // a read run: starts with a request when RUN slots are free, ends after RUN chunks
        if (ld_iss) ds_rc <= (32'(ds_rc) == RUN - 1) ? '0 : ds_rc + 1'b1;
        ds_rr <= ds_rr_nx || (ld_iss && !ds_rr && RUN > 1 && cleft != 1);
        if (pe && o_v) begin
          ob[ds_ocnt % W][ds_ocnt / W] <= o_d;
          ds_ocnt <= ds_ocnt + 1'b1;
        end
        if (ds_run && ds_flushed && og == 0) begin
          ds_run <= 1'b0;
          ds_out <= 1'b1;
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
      end else if (adv) begin
        if (st_pend) begin
          if (pl) cbm <= '0;                    // the chunk's write is taken this cycle
          else
            for (int l = 0; l < W; l++) begin
              cb[pp * W + l] <= t_rdata[l];
              cbm[pp * W + l] <= pm[l];
            end
        end
        st_pend <= st_rd;
        pp <= pos_of(sw);
        pm <= sm;
        pc <= sw & ~32'(CW - 1);
        pl <= seg_end;
        if (st_pend && sleft == 0) begin        // the last write is taken this cycle
          st_pend <= 1'b0;
          ackw <= 1'b1;
        end
      end
    end
  end
endmodule
