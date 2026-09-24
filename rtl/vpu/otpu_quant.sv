// Quantizer for QACT (TMEM rows -> ACT RAM) and QST (TMEM rows -> DRAM bytes), docs/isa.md.
// Per group (a D-block, or a whole row in ROW mode): amax = max|x| over the group (x after the
// optional RSCALE/CSCALE factors), s = amax/127, inv = 127*recip(amax), q = q8(x*inv).
//
// Pipelined for the FPGA clock:
//   read | prescale: (x*r)*c (2 multipliers) | amax tree | scale unit (recip: 3 Newton steps
//   on multiply-add slots, then *127 and *1/127) | quantize: x*inv, q8 | write
// Per-block QACT (the common case) streams: each block is read once into one of NB block
// buffers while its amax folds; the scale unit takes one amax per cycle, and the writer
// quantizes the buffers in order. ROW-mode QACT and QST read each group twice (pass 0 finds
// amax, pass 1 quantizes); QST writes one byte per cycle to DRAM port A, preceded by the
// group's scale word.
// Everything advances only on cycles the TMEM grant is given.

// N-cycle delay line (N >= 2) like otpu_delay, but its last stage has a synchronous reset: a
// flip-flop with a reset is not packed into the shift-register LUTs in front of it, so the
// multiplier operands it feeds leave a real flip-flop and not an SRL's slow clock-to-out.
module otpu_qdly #(parameter int W = 32, parameter int N = 2) (
  input  logic         clk,
  input  logic         rst,
  input  logic         en,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  initial if (N < 2) $fatal(1, "otpu_qdly: N must be at least 2");
  logic [W-1:0] r [N - 1];
  (* shreg_extract = "no", keep *) logic [W-1:0] qr;
  always_ff @(posedge clk) if (en) begin
    r[0] <= d;
    for (int k = 1; k < N - 1; k++) r[k] <= r[k-1];
  end
  always_ff @(posedge clk)
    if (rst) qr <= '0;
    else if (en) qr <= r[N-2];
  assign q = qr;
endmodule

module otpu_qscale
  import otpu_fp::*;
#(parameter int LM = 2, parameter int LA = 4, parameter int TW = 2) (
  input  logic          clk,
  input  logic          rst,
  input  logic          en,
  input  logic          iv,
  input  logic [TW-1:0] itag,
  input  f32_t          amax,        // >= 0 (an absolute value)
  output logic          ov,
  output logic [TW-1:0] otag,
  output f32_t          sc,
  output f32_t          inv
);
  localparam int SL = LM + LA;
  localparam int P = SL + LM;               // one Newton step: u_t, then u_y
  localparam int LAT = 3 * P + LM;
  // input register (the amax selection in front of the unit is its own pipeline stage)
  f32_t          amax_q;
  logic          iv_q;
  logic [TW-1:0] itag_q;
  always_ff @(posedge clk) if (en) begin
    amax_q <= amax; iv_q <= iv; itag_q <= itag;
  end
  // seed stage: |amax|, its flags and the reciprocal seed, registered (amax is an otpu_fmul
  // result or +0, so it is already flushed: no ftz)
  f32_t ax, y0;
  logic zero, big;
  always_ff @(posedge clk) if (en) begin
    f32_t a;
    a = {1'b0, amax_q[30:0]};
    ax <= a;
    zero <= (a == 0);
    big <= (a >= 32'h7E80_0000);
    y0 <= ftz(RECIP_MAGIC - a);
  end
  // recip(ax): y = y * (2 - ax*y), three times, from the magic seed; the delay lines into the
  // multiplier operands end in a flip-flop with a reset (otpu_qdly). The ISA's y*t + (-0) is
  // y*t for every product (no fp_mul result is changed by adding -0), so u_y is a plain multiply,
  // not padded: its output register feeds the next multipliers directly (no SRL in front of
  // their DSPs), and k1 is delayed by the step's SL + LM to meet it.
  f32_t y [4], t [3], k1 [4], yd [3];
  assign y[0] = y0;
  assign k1[0] = {1'b1, ax[30:0]};
  for (genvar i = 0; i < 3; i++) begin : g_it
    otpu_fmadd #(.LM(LM), .LA(LA)) u_t (.clk, .en, .a(k1[i]), .b(y[i]), .c(F_TWO), .y(t[i]));
    otpu_qdly #(.W(32), .N(SL)) u_yd (.clk, .rst, .en, .d(y[i]), .q(yd[i]));
    otpu_qdly #(.W(32), .N(P)) u_k (.clk, .rst, .en, .d(k1[i]), .q(k1[i + 1]));
    otpu_fmul #(.LAT(LM)) u_y (.clk, .en, .a(yd[i]), .b(t[i]), .y(y[i + 1]));
  end
  // big: 127 * (+0) = +0, forced on the product (bd delayed beside zd) instead of on the operand
  f32_t invm, scm, scd;
  logic zd, bd, zd2, bd2;
  otpu_qdly #(.W(2), .N(3 * P)) u_f (.clk, .rst, .en, .d({zero, big}), .q({zd, bd}));
  otpu_fmul #(.LAT(LM)) u_inv (.clk, .en, .a(F_127), .b(y[3]), .y(invm));
  otpu_fmul #(.LAT(LM)) u_sc (.clk, .en, .a(ax), .b(F_INV127), .y(scm));
  otpu_delay #(.W(32), .N(3 * P)) u_scd (.clk, .en, .d(scm), .q(scd));
  otpu_delay #(.W(2), .N(LM)) u_z2 (.clk, .en, .d({zd, bd}), .q({zd2, bd2}));
  otpu_delay #(.W(1 + TW), .N(LAT + 1)) u_v (.clk, .en, .d({iv_q, itag_q}), .q({ov, otag}));
  assign inv = (zd2 || bd2) ? F_ZERO : invm;
  assign sc = zd2 ? F_ZERO : scd;
endmodule

module otpu_quant
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int D     = 32,
  parameter int LANES = 8,
  parameter int SID   = 0
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    start,
  input  cmd_t                    cmd,
  output logic                    rdy,
  output logic                    done,
  input  logic                    gnt,        // TMEM grant: hold everything when low
  // TMEM read (port A lanes)
  output logic [LANES-1:0]        t_ren,
  output logic [LANES-1:0][31:0]  t_raddr,
  input  logic [LANES-1:0][31:0]  t_rdata,
  output logic [LANES-1:0]        t_ren2,
  output logic [LANES-1:0][31:0]  t_raddr2,
  input  logic [LANES-1:0][31:0]  t_rdata2,
  output logic                    t_ren3,     // RSCALE: the row factor (one word)
  output logic [31:0]             t_raddr3,
  input  logic [31:0]             t_rdata3,
  // ACT RAM write
  output logic [LANES-1:0]        act_we,
  output logic [7:0]              act_row,
  output logic [31:0]             act_idx,
  output logic [LANES-1:0][7:0]   act_data,
  output logic                    asc_we,
  output logic [7:0]              asc_row,
  output logic [15:0]             asc_blk,
  output logic [31:0]             asc_data,
  // DRAM port A (QST writes). a_want: a write is waiting (before the grant); the slice withholds
  // the grant while the port cannot take it. QST completes once its writes are acknowledged.
  output logic                    a_want,
  input  logic                    wr_idle,
  output logic                    a_req,
  output logic                    a_we,
  output logic [31:0]             a_addr,
  output logic [31:0]             a_wdata,
  output logic [3:0]              a_be
);
  localparam int LM = 2, LA = 4;
  localparam int NB = 4;                    // block buffers (streaming QACT)
  localparam int BI = $clog2(NB);
  localparam int NC = D / LANES;            // chunks per block
  localparam int QL = 2 * LM;               // prescale latency
  localparam f32_t F_NZ = 32'h8000_0000;
  initial if (D % LANES != 0) $fatal(1, "otpu_quant: LANES must divide D");

  wire en = gnt;

  // ------------------------------------------------------------------ command
  logic        busy, is_st, rowm, csf, rsf, strm, ackw;
  logic [31:0] csb, rsb, sdst, srs, drs, es;
  logic [15:0] rows, KB;
  logic [7:0]  ab;
  logic [31:0] G;                            // elements per group
  logic [31:0] groups;                       // rows * groups per row
  logic [31:0] cyc, st_frz;
  assign rdy = !busy;

  // ------------------------------------------------------------------ reader
  // walks rows r, groups g (block mode: blocks of the row; ROW mode: the row), elements e
  logic [15:0] r;
  logic [31:0] g, e, gi;                     // group in row, element in group, group index
  logic [31:0] row_src, grp_src;             // TMEM address of the row / of the group
  logic [31:0] rel;                          // element index within the row of e
  logic [31:0] badr, bgrp, brs;              // QST byte address dst + r*drs + rel*es (mod 2^32)
                                             // at (r, rel) / the group's start / rel = 0
  logic        pass;                         // two-pass: 0 amax, 1 quantize
  logic        rd_wait;                      // two-pass: waiting for the scale
  logic        rd_done;
  logic [1:0]  bst [NB];                     // 0 free, 1 filling, 2 full (awaiting scale), 3 scaled
  localparam logic [1:0] B_FREE = 2'd0, B_FILL = 2'd1, B_FULL = 2'd2, B_SCL = 2'd3;
  wire [BI-1:0] rbuf = gi[BI-1:0];
  wire [31:0] ew = is_st ? 32'd1 : 32'(LANES);
  wire rlast = (e + ew >= G);
  wire riss = busy && !rd_done && !rd_wait &&
              (!strm || (e != 0) || (bst[rbuf] == B_FREE));

  typedef struct packed {
    logic             v;
    logic             pass;
    logic [BI-1:0]    buf_;
    logic [31:0]      e;         // element offset within the group
    logic [31:0]      rel;       // element index within the row
    logic             last;      // last chunk of the group
    logic             glast;     // ... and of the instruction
    logic [15:0]      row;
    logic [31:0]      grp;       // group within the row
    logic [LANES-1:0] mask;
  } rm_t;
  rm_t m0, mp;                               // data arriving now / after the prescale

  always_comb begin
    t_ren = '0; t_raddr = '0; t_ren2 = '0; t_raddr2 = '0; t_ren3 = 1'b0; t_raddr3 = '0;
    if (riss) begin
      for (int l = 0; l < LANES; l++) begin
        if (32'(l) < ew && e + 32'(l) < G) begin
          t_ren[l] = 1'b1;
          t_raddr[l] = grp_src + e + 32'(l);
          if (csf) begin
            t_ren2[l] = 1'b1;
            t_raddr2[l] = csb + rel + 32'(l);
          end
        end
      end
      if (rsf) begin
        t_ren3 = 1'b1;
        t_raddr3 = rsb + 32'(r);
      end
    end
  end

  // ------------------------------------------------------------------ prescale
  f32_t xp [LANES];
  for (genvar l = 0; l < LANES; l++) begin : g_pre
    f32_t v1, cd;
    otpu_fmul #(.LAT(LM)) u_r (.clk, .en, .a(t_rdata[l]), .b(rsf ? t_rdata3 : F_ONE), .y(v1));
    otpu_delay #(.W(32), .N(LM)) u_c (.clk, .en, .d(csf ? t_rdata2[l] : F_ONE), .q(cd));
    otpu_fmul #(.LAT(LM)) u_c2 (.clk, .en, .a(v1), .b(cd), .y(xp[l]));
  end
  otpu_delay #(.W($bits(rm_t)), .N(QL)) u_mp (.clk, .en, .d(m0), .q(mp));

  // two-pass QST byte address dst + row*drs + rel*es: the reader's running sum badr, taken in
  // the issue cycle like m0 (one register) and delayed with it through the prescale (QL)
  logic [31:0] mp_baddr;
  otpu_delay #(.W(32), .N(QL + 1)) u_ba (.clk, .en, .d(badr), .q(mp_baddr));

  // fp_gt for the amax path: every operand there is a non-negative flushed value (an otpu_fmul
  // result with the sign cleared, +0, or one taken from those), where fkey is {1, a[30:0]}, so
  // the order is the unsigned order of a[30:0] (NaN still above +inf) and needs no ftz
  function automatic logic mgt(input f32_t a, input f32_t b);
    return a[30:0] > b[30:0];
  endfunction

  // chunk amax (tree over the lanes) in two registered stages: the first level, then the rest
  localparam int HL = LANES / 2;
  f32_t  ch [HL];
  f32_t  cmax;
  rm_t   mh, mc;
  always_ff @(posedge clk) if (en) begin
    f32_t v [LANES];
    // xp is flushed (otpu_fmul), so clearing the sign is fabs
    for (int l = 0; l < LANES; l++) v[l] = mp.mask[l] ? {1'b0, xp[l][30:0]} : F_ZERO;
    for (int l = 0; l < HL; l++) ch[l] <= mgt(v[l + HL], v[l]) ? v[l + HL] : v[l];
    mh <= mp;
  end
  always_ff @(posedge clk) if (en) begin
    f32_t v [HL];
    for (int l = 0; l < HL; l++) v[l] = ch[l];
    for (int w = HL / 2; w >= 1; w = w / 2)
      for (int l = 0; l < w; l++) if (mgt(v[l + w], v[l])) v[l] = v[l + w];
    cmax <= v[0];
    mc <= mh;
  end

  // block buffers (streaming QACT): chunk c of buffer b holds elements c*LANES .. (the data
  // itself is in per-lane RAMs beside the writer, g_sb)
  f32_t        bamax [NB];
  logic [15:0] brow [NB];
  logic [31:0] bblk [NB];
  f32_t        bsc [NB], binv [NB];

  // the group's running amax with this chunk folded in (a group's first chunk restarts it). The
  // two-pass mode keeps its pass-0 max in bamax[mc.buf_] too: buf_ is constant over a group.
  f32_t        amx, nmax;
  assign amx  = bamax[mc.buf_];
  assign nmax = (mc.e != 0 && mgt(amx, cmax)) ? amx : cmax;

  // scale unit (its amax is don't-care while sq_iv is low: sc/inv are read only under sq_ov)
  logic        sq_iv, sq_ov;
  logic [BI-1:0] sq_itag, sq_otag;
  f32_t        sq_amax, sq_sc, sq_inv;
  assign sq_iv   = mc.v && mc.last && (strm || !mc.pass);
  assign sq_itag = mc.buf_;
  assign sq_amax = nmax;
  otpu_qscale #(.LM(LM), .LA(LA), .TW(BI)) u_sc (.clk, .rst, .en, .iv(sq_iv), .itag(sq_itag),
    .amax(sq_amax), .ov(sq_ov), .otag(sq_otag), .sc(sq_sc), .inv(sq_inv));
  f32_t sc2, inv2;                           // two-pass: the current group's scale

  // ------------------------------------------------------------------ writer
  // streaming: buffers in order; two-pass: pass-1 chunks from the prescale output
  logic [BI-1:0] wsel;
  logic [31:0]   wc;                         // chunk of the buffer being written
  logic [31:0]   wdone;                      // groups written (streaming)
  wire w_go = busy && strm && bst[wsel] == B_SCL;

  // block buffer data: one RAM per lane, (buffer, chunk) -> word, with one write port (the
  // prescale output, under the sequencer's condition) and one asynchronous read port (the writer)
  localparam int CI = (NC > 1) ? $clog2(NC) : 1;
  wire [BI+CI-1:0] sb_wa = {mp.buf_, CI'(mp.e / LANES)};
  wire [BI+CI-1:0] sb_ra = {wsel, CI'(wc)};
  wire sb_we = !rst && !ackw && !start && busy && en && mp.v && strm;
  f32_t sb_rd [LANES];
  for (genvar l = 0; l < LANES; l++) begin : g_sb
    f32_t mem [NB * (2 ** CI)];
    always_ff @(posedge clk) if (sb_we) mem[sb_wa] <= xp[l];
    assign sb_rd[l] = mem[sb_ra];
  end

  typedef struct packed {
    logic             v;
    logic [LANES-1:0] mask;
    logic [7:0]       row;
    logic [31:0]      idx;       // ACT byte index
    logic             sw;        // write the scale (first chunk of a block)
    logic [15:0]      blk;
    logic             st;        // QST byte
    logic [31:0]      baddr;     // QST byte address
    logic             fin;       // last write of the instruction
  } wm_t;
  wm_t   wq, wqd;                            // at the multiplier input / at the write stage
  f32_t  wx [LANES];
  f32_t  winv, wsc_in, wsc_d;
  always_comb begin
    wq = '0;
    winv = F_ZERO;
    wsc_in = F_ZERO;
    for (int l = 0; l < LANES; l++) wx[l] = F_ZERO;
    if (w_go) begin
      wq.v = 1'b1;
      for (int l = 0; l < LANES; l++) begin
        wq.mask[l] = 1'b1;
        wx[l] = sb_rd[l];
      end
      wq.row = brow[wsel][7:0];
      wq.idx = (32'(ab) + bblk[wsel]) * D + wc * LANES;
      wq.sw = (wc == 0);
      wq.blk = 16'(ab) + 16'(bblk[wsel]);
      wq.fin = (wc + 1 == NC) && (wdone + 1 == groups);
      winv = binv[wsel];
      wsc_in = bsc[wsel];
    end else if (!strm && mp.v && mp.pass) begin
      wq.v = 1'b1;
      wq.mask = mp.mask;
      for (int l = 0; l < LANES; l++) wx[l] = xp[l];
      wq.row = mp.row[7:0];
      wq.idx = 32'(ab) * D + mp.rel;
      wq.sw = !is_st && (mp.rel % D == 0);
      wq.blk = 16'(ab) + 16'(mp.rel / D);
      wq.st = is_st;
      wq.baddr = mp_baddr;
      wq.fin = mp.glast;
      winv = inv2;
      wsc_in = sc2;
    end
  end
  // the writer's selection registered in front of the multipliers
  wm_t   wq_r;
  f32_t  wx_r [LANES];
  f32_t  winv_r, wsc_r;
  always_ff @(posedge clk) if (en) begin
    wq_r <= wq;
    wx_r <= wx;
    winv_r <= winv;
    wsc_r <= wsc_in;
  end
  // q8_s1 with the shift (and its guard bit k = sh - 1) decoded from e by a table, not computed
  // as 150 - e: no carry chain between e and ip/g/st (the e compares feed only zero/sat). No
  // ftz: it changes only m, and only when e == 0, where zero makes q8_s2 return 0 anyway.
  function automatic q8_mid_t qq8_s1(input f32_t x_in);
    q8_mid_t q;
    f32_t x;
    logic [7:0]  e;
    logic [23:0] m;
    logic [4:0]  sh, k;
    x = x_in;
    e = x[30:23];
    m = {1'b1, x[22:0]};
    q = '0;
    q.s = x[31];
    q.zero = (e < 8'd126);
    q.sat = (e >= 8'd134);
    sh = 5'd17; k = 5'd16;                      // zero or sat (as q8_s1)
    for (int c = 126; c < 134; c++)
      if (e == 8'(c)) begin sh = 5'(150 - c); k = 5'(149 - c); end
    q.ip = 32'(m) >> sh;
    q.g = m[k];
    q.st = |(m & ~(24'hFF_FFFF << k));          // |m[k-1:0]
    return q;
  endfunction
  logic [LANES-1:0][7:0] qb;
  for (genvar l = 0; l < LANES; l++) begin : g_q
    f32_t p;
    q8_mid_t qm;
    otpu_fmul #(.LAT(LM)) u_q (.clk, .en, .a(wx_r[l]), .b(winv_r), .y(p));
    always_ff @(posedge clk) if (en) begin
      qm <= qq8_s1(p);
      qb[l] <= q8_s2(qm);
    end
  end
  otpu_delay #(.W($bits(wm_t)), .N(LM + 2)) u_wq (.clk, .en, .d(wq_r), .q(wqd));
  otpu_delay #(.W(32), .N(LM + 2)) u_ws (.clk, .en, .d(wsc_r), .q(wsc_d));

  // QST scale word: written when the group's scale arrives (no data writes are in flight)
  logic [31:0] saddr;                         // next scale word address
  wire  st_scale_w = busy && !ackw && is_st && sq_ov;
  // a DRAM write is waiting (independent of the grant, which the slice derives from it)
  assign a_want = (busy && !ackw && wqd.v && wqd.st) || st_scale_w;

  always_comb begin
    act_we = '0; act_row = '0; act_idx = '0; act_data = '0;
    asc_we = 1'b0; asc_row = '0; asc_blk = '0; asc_data = '0;
    a_req = 1'b0; a_we = 1'b0; a_addr = '0; a_wdata = '0; a_be = '0;
    if (busy && !ackw && wqd.v) begin
      if (wqd.st) begin
        a_req = 1'b1;
        a_we = 1'b1;
        a_addr = wqd.baddr >> 2;
        a_wdata = {4{qb[0]}};
        a_be = 4'b0001 << wqd.baddr[1:0];
      end else begin
        act_row = wqd.row;
        act_idx = wqd.idx;
        for (int l = 0; l < LANES; l++) begin
          act_we[l] = wqd.mask[l];
          act_data[l] = qb[l];
        end
        if (wqd.sw) begin
          asc_we = 1'b1;
          asc_row = wqd.row;
          asc_blk = wqd.blk;
          asc_data = wsc_d;
        end
      end
    end
    if (st_scale_w) begin
      a_req = 1'b1;
      a_we = 1'b1;
      a_addr = saddr >> 2;
      a_wdata = sq_sc;
      a_be = 4'hF;
    end
    if (!gnt) begin
      act_we = '0; asc_we = 1'b0; a_req = 1'b0; a_we = 1'b0;
    end
  end

  // ------------------------------------------------------------------ sequencing
  always_ff @(posedge clk) begin
    done <= 1'b0;
    cyc <= cyc + 1;
    if (busy && !gnt) st_frz <= st_frz + 1;
    if (rst) begin
      busy <= 1'b0;
      ackw <= 1'b0;
      m0 <= '0;
      cyc <= '0;
    end else if (ackw) begin
      m0 <= '0;
      if (wr_idle) begin
        ackw <= 1'b0;
        busy <= 1'b0;
        done <= 1'b1;
      end
    end else if (start) begin
      logic qst;
      qst = (cmd.op == OP_QST);
      is_st <= qst;
      rowm  <= cmd.flags[0];
      csf   <= !qst && cmd.flags[1];
      csb   <= cmd.w4;
      rsf   <= !qst && cmd.flags[2];
      rsb   <= cmd.w5;
      strm  <= !qst && !cmd.flags[0];
      if (qst) begin
        sdst <= cmd.w3;
        rows <= cmd.w4[15:0];
        KB   <= cmd.w4[31:16];
        srs  <= cmd.w5;
        drs  <= cmd.w6;
        es   <= cmd.w7;
        G    <= cmd.flags[0] ? 32'(cmd.w4[31:16]) * D : D;
        groups <= 32'(cmd.w4[15:0]) * (cmd.flags[0] ? 32'd1 : 32'(cmd.w4[31:16]));
      end else begin
        rows <= 16'(cmd.w2[7:0]);
        ab   <= cmd.w2[15:8];
        KB   <= cmd.w2[31:16];
        srs  <= cmd.w3;
        G    <= cmd.flags[0] ? 32'(cmd.w2[31:16]) * D : D;
        groups <= 32'(cmd.w2[7:0]) * (cmd.flags[0] ? 32'd1 : 32'(cmd.w2[31:16]));
      end
      r <= '0; g <= '0; e <= '0; gi <= '0; rel <= '0;
      badr <= cmd.w2; bgrp <= cmd.w2; brs <= cmd.w2;     // QST dst (unused by QACT)
      row_src <= cmd.w1; grp_src <= cmd.w1;
      pass <= 1'b0; rd_wait <= 1'b0;
      saddr <= cmd.w3;
      for (int b = 0; b < NB; b++) bst[b] <= B_FREE;
      wsel <= '0; wc <= '0; wdone <= '0;
      m0 <= '0;
      st_frz <= '0;
      if ((qst ? cmd.w4[15:0] : 16'(cmd.w2[7:0])) == 0 || (qst ? cmd.w4[31:16] : cmd.w2[31:16]) == 0) begin
        rd_done <= 1'b1;
        done <= 1'b1;
      end else begin
        rd_done <= 1'b0;
        busy <= 1'b1;
      end
    end else if (busy && en) begin
      logic fin;
      fin = 1'b0;
      // ---- reader
      m0 <= '0;
      if (riss) begin
        m0.v <= 1'b1;
        m0.pass <= pass;
        m0.buf_ <= rbuf;
        m0.e <= e;
        m0.rel <= rel;
        m0.last <= rlast;
        m0.row <= r;
        m0.grp <= g;
        for (int l = 0; l < LANES; l++) m0.mask[l] <= (32'(l) < ew && e + 32'(l) < G);
        m0.glast <= rlast && (gi + 1 == groups);
        if (strm && e == 0) begin
          bst[rbuf] <= B_FILL;
          brow[rbuf] <= r;
          bblk[rbuf] <= g;
        end
        // badr follows rel (QST: ew = 1, so rel + ew is badr + es)
        if (!rlast) begin
          e <= e + ew;
          rel <= rel + ew;
          badr <= badr + es;
        end else if (!strm && !pass) begin        // two-pass: wait for the scale, re-read
          rd_wait <= 1'b1;
          e <= '0;
          rel <= rel - e;
          badr <= bgrp;
        end else begin                             // next group
          e <= '0;
          gi <= gi + 1;
          pass <= 1'b0;
          if (gi + 1 == groups) rd_done <= 1'b1;
          if (G == D && g + 1 < 32'(KB)) begin
            g <= g + 1;
            rel <= rel + ew;
            badr <= badr + es;
            bgrp <= badr + es;
            grp_src <= grp_src + D;
          end else begin
            g <= '0;
            rel <= '0;
            r <= r + 1;
            brs <= brs + drs;
            badr <= brs + drs;
            bgrp <= brs + drs;
            row_src <= row_src + srs;
            grp_src <= row_src + srs;
          end
        end
      end
      // ---- after the prescale: buffers (the data: g_sb) and amax
      if (mc.v && (strm || !mc.pass)) bamax[mc.buf_] <= nmax;
      if (mc.v && strm && mc.last) bst[mc.buf_] <= B_FULL;
      // ---- scale results
      if (sq_ov) begin
        if (strm) begin
          bsc[sq_otag] <= sq_sc;
          binv[sq_otag] <= sq_inv;
          bst[sq_otag] <= B_SCL;
        end else begin
          sc2 <= sq_sc;
          inv2 <= sq_inv;
          pass <= 1'b1;
          rd_wait <= 1'b0;
          if (is_st) saddr <= saddr + 4;
        end
      end
      // ---- streaming writer
      if (w_go) begin
        if (wc + 1 == NC) begin
          wc <= '0;
          bst[wsel] <= B_FREE;
          wsel <= wsel + 1;
          wdone <= wdone + 1;
        end else begin
          wc <= wc + 1;
        end
      end
      // ---- the last write
      if (wqd.v && wqd.fin) fin = 1'b1;
      if (fin && is_st) begin
        ackw <= 1'b1;                              // the last write is taken this cycle
      end else if (fin) begin
        busy <= 1'b0;
        done <= 1'b1;
`ifndef SYNTHESIS
        if (trace) $display("T%0d U c=%0d u=2 frz=%0d", SID, cyc, st_frz);
`endif
      end
    end
  end
`ifndef SYNTHESIS
  bit trace;
  initial trace = $test$plusargs("trace");
`endif
endmodule
