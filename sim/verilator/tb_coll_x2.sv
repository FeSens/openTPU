// The collective across two cards, alone: on each card otpu_coll behind otpu_coll_hub at NC 2
// (ncards 2, CARD_ID 0 and 1), its S = 2 slices XRS register stages away each way (otpu_xr, as
// otpu_board g_xs), each with its otpu_coll_port and a model of its TMEM's collective port (the
// grant withheld at random: +stall=P percent, runs up to +burst=B), and the hub's two links each
// through its crossing (otpu_coll_lx) to a cable joined to the other card's: port p to port p,
// or crossed (CROSS: port 0 to the other card's port 1). The cable is qsfp-link's otpu_link_sim
// (LSIM 1: its plusargs, +lnk<p>_L= the latency, +lnk<p>a_flip= / _drop= / _down= / _stall=
// ...; end a card 0's) or otpu_link_stub (LSIM 0: +xlat=, +xstall= the senders' tready withheld
// at random, +xekind=seq a header's collective number changed). Global slice g = 2 c + k (card
// c's slice k), each with its own sources. Each card has its run's reset (RUN 0: the hub's rst,
// the slices') and both share the links' reset (the hub's lrst, the crossings', the cables'):
// card 1's run can start +c1late=N cycles after card 0's, its links up all along.
// Both cards walk one list of +ops=N collectives, GATHERs of random rows, cols and strides (the
// sources in the low half of TMEM, the destinations in the high half; one row at times one run,
// seg = cols: FOLLOW's shape) and BARs, each card on its own (+gap=G idle cycles between, up to
// +cskew=K more on card 1 each time, +c1wait=W more always; the slices of a card up to +skew=K
// apart), so a card runs up to a collective ahead and the other card's words for it wait.
// +big: GATHERs of one row of +bigcols=C words a slice (20,480: 2,560 words a piece, a link's
// RX FIFO is 2,048) instead (TW 2^18). Checked on every slice: its writes of each piece (g) in
// order and as the reference's (each piece's walk: rows, LANES words), every piece's in before
// the slice's ack, the destination holding all four sources' words after each GATHER; FOLLOW: a
// write the hub tags (coll_wtop) belongs to the local half, in order; no hang (the cards'
// collectives all end). TBCX DRAIN: the cycles a card's hub took another card's write while its
// TX held its own writes back (the links' independence). +wd=W: the hubs' watchdog (cycles).
// With an error injected a card aborts and its collectives end on their local writes: its
// checks then stop, and the run must still end, the cards' X2_HUB printed (TBCX STATUS); with a
// flipped word (+xekind=flip / +lnk*_flip) the writes' words are not checked, their lanes and
// addresses are. +sizes: GATHERs of one row of 16, 256 and 4,096 words a slice, +reps=R each,
// timed (TBCX SIZE: the cycles from the previous one's end to the later card's ack). +seed=N.
module tb_coll_x2;
  import otpu_pkg::*;
  parameter int XRS = 2;                   // otpu_board XRS
  parameter int TW  = 1 << 16;             // TMEM words a slice
  parameter bit CROSS = 1'b0;              // card 0's port p to card 1's port 1 - p
  parameter bit LSIM = 1'b1;               // the cable: otpu_link_sim (1) or otpu_link_stub (0)
  localparam int S = 2, NCD = 2, G = S * NCD, L = 8;
  localparam int NR = 2 ** $clog2(2 * XRS + 8);       // (otpu_board CNR, CF: COLL_PIPE 1)
  localparam int F  = 2 ** $clog2(2 * XRS + 6);
  localparam int HW = TW / 2;

  logic clk = 1'b0;
  always #5 clk = ~clk;
  longint cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  int seed = 1, nops = 200, stall = 20, burst = 6, skew = 6, cskew = 200, gap = 4, reps = 4;
  int wd = 0, c1late = 0, c1wait = 0, bigcols = 20480;
  bit sizes = 1'b0, big = 1'b0, nodata = 1'b0;   // nodata: a flipped word (its data unchecked)
  initial begin
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("ops=%d", nops));
    void'($value$plusargs("stall=%d", stall));
    void'($value$plusargs("burst=%d", burst));
    void'($value$plusargs("skew=%d", skew));
    void'($value$plusargs("cskew=%d", cskew));
    void'($value$plusargs("gap=%d", gap));
    void'($value$plusargs("reps=%d", reps));
    void'($value$plusargs("wd=%d", wd));
    void'($value$plusargs("c1late=%d", c1late));
    void'($value$plusargs("c1wait=%d", c1wait));
    void'($value$plusargs("bigcols=%d", bigcols));
    sizes = $test$plusargs("sizes");
    big = $test$plusargs("big");
    nodata = $test$plusargs("xekind=flip") || $test$plusargs("lnk0a_flip") ||
             $test$plusargs("lnk0b_flip") || $test$plusargs("lnk1a_flip") ||
             $test$plusargs("lnk1b_flip");
    void'($urandom(seed));
  end

  logic lrst = 1'b1;                       // the links'
  logic [NCD-1:0] crst = '1;               // each card's run's

  // ---- the collectives, the same list for both cards
  typedef struct {
    bit bar;
    int rows, cols, drs, seg, dst;
    int w1 [G];
    int w4 [G];
  } op_t;
  op_t ops [$];
  logic [31:0] src0 [G][HW];               // each global slice's sources (never written)
  initial for (int g = 0; g < G; g++) for (int i = 0; i < HW; i++) src0[g][i] = $urandom;
  function automatic int rnd(int lo, int hi);
    return lo + int'($urandom_range(hi - lo));
  endfunction
  function automatic op_t mk_gather(int rows, int cols);
    op_t o;
    int span;
    o.bar = 1'b0;
    o.rows = rows;
    o.cols = cols;
    for (int g = 0; g < G; g++) begin
      o.w4[g] = cols + rnd(0, 8) * (rows > 1);
      o.w1[g] = rnd(0, HW - ((rows - 1) * o.w4[g] + cols));
    end
    o.drs = cols + rnd(0, 8) * (rows > 1);
    o.seg = (rows - 1) * o.drs + cols + ((rows == 1 && rnd(0, 1) == 1) ? 0 : rnd(0, 16));
    span = (G - 1) * o.seg + (rows - 1) * o.drs + cols;
    o.dst = HW + rnd(0, HW - span);
    return o;
  endfunction
  function automatic op_t mk_bar();
    op_t o;
    o = '{default: 0};
    o.bar = 1'b1;
    return o;
  endfunction

  // ---- the two cards
  logic    [NCD-1:0] fin;                  // each card walked the whole list
  lk_in_t  li [NCD][2];
  lk_out_t lo [NCD][2];
  for (genvar c = 0; c < NCD; c++) begin : g_c
    logic coll_rst;
    always_ff @(posedge clk) coll_rst <= crst[c];

    // the unit and the hub
    logic [S-1:0]                coll_req;
    logic                        coll_ack, coll_gnt;
    cmd_t                        coll_cmds [S];
    logic [S-1:0][L-1:0]         coll_ren;
    logic [S-1:0][L-1:0][31:0]   coll_raddr, coll_rdata;
    logic [L-1:0]                coll_wen;
    logic [L-1:0][31:0]          coll_waddr, coll_wdata;
    otpu_coll #(.S(S), .LANES(L), .RQ(1'b0)) u_coll (
      .clk, .rst(coll_rst), .req(coll_req), .cmds(coll_cmds), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata));

    logic                        h_ack, h_wv, h_wt;
    logic [S-1:0]                h_rv, h_req, h_dv, h_pop;
    logic [L-1:0]                h_rm, h_wm;
    logic [L-1:0][31:0]          h_ra, h_wa, h_wd;
    logic [S-1:0][L-1:0][31:0]   h_dd;
    cmd_t                        h_cmd [S];
    logic [1:0]                  l_tv, l_tlast, l_rcred, l_tcred, l_rv, l_rhdr, l_rlast;
    logic [1:0]                  l_up, l_rerr, l_abin;
    logic [1:0][255:0]           l_td, l_rd;
    logic                        l_abort, xerr;
    logic [7:0]                  xst;
    logic [15:0]                 xseq;
    otpu_coll_hub #(.S(S), .LANES(L), .F(F), .NR(NR), .NC(NCD)) u_chub (
      .clk, .rst(coll_rst),
      .u_req(coll_req), .u_cmds(coll_cmds), .u_gnt(coll_gnt), .u_ack(coll_ack),
      .u_ren(coll_ren), .u_raddr(coll_raddr), .u_rdata(coll_rdata),
      .u_wen(coll_wen), .u_waddr(coll_waddr), .u_wdata(coll_wdata),
      .x_req(h_req), .x_cmds(h_cmd), .x_ack(h_ack), .x_rv(h_rv), .x_rm(h_rm), .x_ra(h_ra),
      .x_dv(h_dv), .x_dd(h_dd), .x_wv(h_wv), .x_wm(h_wm), .x_wa(h_wa), .x_wd(h_wd),
      .x_pop(h_pop), .x_wt(h_wt), .lrst, .cid(8'(c)), .ncards(8'(NCD)), .wdog(32'(wd)),
      .l_tv, .l_tlast, .l_td, .l_rcred, .l_abort, .l_tcred, .l_rv, .l_rhdr, .l_rlast, .l_rd,
      .l_up, .l_rerr, .l_abin, .xst, .xerr, .xseq);
    for (genvar p = 0; p < 2; p++) begin : g_l
      otpu_coll_lx #(.XRS(XRS)) u_lx (
        .clk, .rst(lrst), .ren(!coll_rst), .t_v(l_tv[p]), .t_last(l_tlast[p]), .t_d(l_td[p]),
        .r_cred(l_rcred[p]), .abort(l_abort), .card(8'(c)), .t_cred(l_tcred[p]), .r_v(l_rv[p]),
        .r_hdr(l_rhdr[p]), .r_last(l_rlast[p]), .r_d(l_rd[p]), .up(l_up[p]), .r_err(l_rerr[p]),
        .ab_in(l_abin[p]), .li(li[c][p]), .lo(lo[c][p]));
    end

    // the links' independence: the hub took another card's write while its TX held back one of
    // its own slices' writes
    longint ndrain = 0, nahead = 0;
    always @(posedge clk)
      if (!coll_rst && u_chub.g_pipe.c_wt && u_chub.g_pipe.c_tok != '1) ndrain <= ndrain + 1;

    // A complete collective ahead: this card has issued k+1 while its peer still names k,
    // and its local TX is held by credits. The large-GATHER test must observe this state.
    always @(posedge clk)
      if (!coll_rst && opn > g_c[1-c].opn && u_chub.g_pipe.c_act &&
          u_chub.g_pipe.c_tok != '1) nahead <= nahead + 1;

    // the collective the drivers ask for now (its index on this card), its writes per piece
    typedef struct packed {
      logic [L-1:0]       m;
      logic [L-1:0][31:0] a, d;
    } wr_t;
    op_t   cur;
    int    opn = 0;
    bit    go = 1'b0, chk = 1'b0, fol;
    wr_t   exl [G][$];                     // piece g's writes, in order (every slice's)
    logic [S-1:0] done;
    int    nwr = 0, gathers = 0, bars = 0, ntag = 0, nfol = 0;
    int abort_op = -1;
    always @(posedge clk) if (abd && abort_op < 0) abort_op <= opn;
    bit    abd;                            // aborted: no more checks on this card
    assign abd = xst[0];
    assign fol = !cur.bar && cur.rows == 1 && cur.seg == cur.cols;

    // each slice: the crossing (XRS stages each way), its port, its TMEM's model, its driver
    logic [S-1:0] x_rst, sc_req, sc_ack, gl;
    cmd_t         sc_cmd [S];
    for (genvar s = 0; s < S; s++) begin : g_s
      localparam int DW = 1 + 1 + 32 * L + 32 * L + L + 32 * L + L + 3;
      localparam int UW = $bits(cmd_t) + 32 * L + 3;
      logic [DW-1:0] dn_d, dn_q;
      logic [UW-1:0] up_d, up_q;
      logic [XRS-2:0][DW-1:0] dn_m;
      logic [XRS-2:0][UW-1:0] up_m;
      logic               rx_crst, p_rv, p_wv, p_wt, p_dv, p_pop, cwtop;
      logic [L-1:0]       p_rm, p_wm, cren, cwen;
      logic [L-1:0][31:0] p_ra, p_wa, p_wd, p_dd, craddr, crdata, cwaddr, cwdata;
      assign dn_d = {crst[c], h_wt, h_wd, h_wa, h_wm, h_ra, h_rm, h_wv, h_rv[s], h_ack};
      otpu_xr #(.W(DW), .RW(3)) u_dn_tx (.clk, .rst(coll_rst), .d(dn_d), .q(dn_m[0]));
      for (genvar i = 1; i < XRS - 1; i++) begin : g_dm
        otpu_xr #(.W(DW), .RW(3)) u_dn (.clk, .rst(coll_rst), .d(dn_m[i-1]), .q(dn_m[i]));
      end
      otpu_xr #(.W(DW), .RW(3)) u_dn_rx (.clk, .rst(rx_crst), .d(dn_m[XRS-2]), .q(dn_q));
      assign {rx_crst, p_wt, p_wd, p_wa, p_wm, p_ra, p_rm, p_wv, p_rv, sc_ack[s]} = dn_q;
      assign x_rst[s] = rx_crst;

      otpu_coll_port #(.LANES(L), .F(F), .NR(NR), .WT(1'b1)) u_cport (
        .clk, .rst(x_rst[s]), .rv(p_rv), .rm(p_rm), .ra(p_ra), .wv(p_wv), .wm(p_wm), .wa(p_wa),
        .wd(p_wd), .wt(p_wt), .dv(p_dv), .dd(p_dd), .pop(p_pop),
        .coll_ren(cren), .coll_raddr(craddr), .coll_rdata(crdata),
        .coll_wen(cwen), .coll_waddr(cwaddr), .coll_wdata(cwdata), .coll_wtop(cwtop),
        .gnt_local(gl[s]), .gnt());

      assign up_d = {sc_cmd[s], p_dd, p_pop, p_dv, sc_req[s]};
      otpu_xr #(.W(UW), .RW(3)) u_up_tx (.clk, .rst(x_rst[s]), .d(up_d), .q(up_m[0]));
      for (genvar i = 1; i < XRS - 1; i++) begin : g_um
        otpu_xr #(.W(UW), .RW(3)) u_up (.clk, .rst(x_rst[s]), .d(up_m[i-1]), .q(up_m[i]));
      end
      otpu_xr #(.W(UW), .RW(3)) u_up_rx (.clk, .rst(coll_rst), .d(up_m[XRS-2]), .q(up_q));
      assign {h_cmd[s], h_dd[s], h_pop[s], h_dv[s], h_req[s]} = up_q;

      // the TMEM's collective port: global slice 2 c + s's sources below HW
      logic [31:0] mem [TW];
      logic [L-1:0][31:0] rq;
      int run_left = 0;
      int ix [G];                          // each piece's next write
      initial for (int i = 0; i < TW; i++) mem[i] = (i < HW) ? src0[S * c + s][i] : 32'hDEAD0000 | i;
      assign crdata = rq;
      always @(posedge clk) begin
        if (run_left > 0) run_left <= run_left - 1;
        else if (int'($urandom_range(99)) < stall) run_left <= int'($urandom_range(burst - 1));
      end
      assign gl[s] = run_left == 0;
      always @(posedge clk) begin
        if (gl[s])
          for (int l = 0; l < L; l++) if (cren[l]) rq[l] <= mem[craddr[l] % TW];
        if (gl[s] && |cwen) begin
          wr_t w, e;
          int g;
          w.m = cwen;
          for (int l = 0; l < L; l++) begin
            w.a[l] = cwen[l] ? cwaddr[l] : '0;
            w.d[l] = cwen[l] ? cwdata[l] : '0;
            if (cwen[l]) mem[cwaddr[l] % TW] <= cwdata[l];
          end
          g = (int'(w.a[0]) - cur.dst) / (cur.seg > 0 ? cur.seg : 1);
          if (abd && abort_op >= 0 && opn > abort_op && g / S != c)
            $fatal(1, "tb_coll_x2: remote write after abort (card %0d op %0d)", c, opn);
          if (!abd || g / S == c) begin
            if (cur.bar || g < 0 || g >= G || ix[g] >= exl[g].size())
              $fatal(1, "tb_coll_x2: card %0d slice %0d wrote %0d (%h) with no write of it expected (op %0d, piece %0d)",
                     c, s, w.a[0], w.m, opn, g);
            e = exl[g][ix[g]];
            if (nodata && g / S != c) begin w.d = '0; e.d = '0; end
            if (w != e)
              $fatal(1, "tb_coll_x2: card %0d slice %0d piece %0d write %0d of op %0d: lanes %h addr %0d data %h, expected lanes %h addr %0d data %h",
                     c, s, g, ix[g], opn, w.m, w.a[0], w.d[0], e.m, e.a[0], e.d[0]);
            // Phase 1 FOLLOW advances only on the local half, in its original order.
            if (cwtop && g / S != c)
              $fatal(1, "tb_coll_x2: remote piece %0d advances card %0d's local FOLLOW", g, c);
            if (fol && cwtop) begin
              for (int h = S * c; h < g; h++)
                if (ix[h] != exl[h].size())
                  $fatal(1, "tb_coll_x2: local FOLLOW advances before piece %0d", h);
              if (s == 0) ntag <= ntag + 1;
            end
            ix[g] <= ix[g] + 1;
          end
          if (s == 0) nwr <= nwr + 1;
        end
        if (!go) for (int g = 0; g < G; g++) ix[g] <= 0;
      end

      // the driver: coll_req from the start until its ack, as otpu_slice's
      int  wait_left = 0;
      bit  asked = 1'b0, dn = 1'b0;
      assign done[s] = dn;
      always @(posedge clk) begin
        if (x_rst[s]) begin
          sc_req[s] <= 1'b0;
          asked <= 1'b0;
        end else if (!go) begin
          asked <= 1'b0;
          dn <= 1'b0;
          wait_left <= int'($urandom_range(skew));
        end else if (!dn && !asked) begin
          if (wait_left == 0) begin
            cmd_t k;
            k = '0;
            k.op = cur.bar ? OP_BAR : OP_GATHER;
            if (!cur.bar) begin
              k.w1 = 32'(cur.w1[S * c + s]);
              k.w2 = 32'(cur.dst);
              k.w3 = {16'(cur.cols), 16'(cur.rows)};
              k.w4 = 32'(cur.w4[S * c + s]);
              k.w5 = 32'(cur.drs);
              k.w6 = 32'(cur.seg);
            end
            sc_req[s] <= 1'b1;
            sc_cmd[s] <= k;
            asked <= 1'b1;
          end else wait_left <= wait_left - 1;
        end else if (sc_req[s] && sc_ack[s]) begin
          sc_req[s] <= 1'b0;
          for (int g = 0; g < G; g++)
              if ((!abd || g / S == c) && ix[g] != exl[g].size())
                $fatal(1, "tb_coll_x2: card %0d slice %0d's ack of op %0d with piece %0d's %0d writes still to come",
                       c, s, opn, g, exl[g].size() - ix[g]);
          dn <= 1'b1;
        end
      end
      initial sc_req[s] = 1'b0;

      // the destination after a GATHER (chk): every source's words
      always @(posedge clk)
        if (chk && !abd && !nodata)
          for (int g = 0; g < G; g++)
            for (int r = 0; r < cur.rows; r++)
              for (int cc = 0; cc < cur.cols; cc++) begin
                int da, sa;
                da = cur.dst + g * cur.seg + r * cur.drs + cc;
                sa = cur.w1[g] + r * cur.w4[g] + cc;
                if (mem[da % TW] != src0[g][sa % HW])
                  $fatal(1, "tb_coll_x2: card %0d slice %0d's word %0d (piece %0d row %0d col %0d) after op %0d: %h, not %h",
                         c, s, da, g, r, cc, opn, mem[da % TW], src0[g][sa % HW]);
              end
    end

    // the reference: each piece's writes (every slice's)
    task automatic expect_op();
      for (int g = 0; g < G; g++) exl[g].delete();
      if (cur.bar) return;
      for (int g = 0; g < G; g++)
        for (int r = 0; r < cur.rows; r++)
          for (int cc = 0; cc < cur.cols; cc += L) begin
            wr_t e;
            e = '0;
            for (int l = 0; l < L; l++)
              if (cc + l < cur.cols) begin
                int sa;
                e.m[l] = 1'b1;
                e.a[l] = 32'(cur.dst + g * cur.seg + r * cur.drs + cc + l);
                sa = cur.w1[g] + r * cur.w4[g] + cc + l;
                e.d[l] = src0[g][sa % HW];
              end
            exl[g].push_back(e);
          end
    endtask
    task automatic run_op(int n, output longint t_end);
      longint t0;
      cur = ops[n];
      opn = n;
      expect_op();
      if (cur.bar) bars++; else gathers++;
      if (!cur.bar && cur.rows == 1 && cur.seg == cur.cols) nfol++;
      @(posedge clk);
      go <= 1'b1;
      t0 = cyc;
      while (done != '1) begin
        @(posedge clk);
        if (cyc - t0 > 4000000) $fatal(1, "tb_coll_x2: card %0d's op %0d hangs (done %b)", c, n, done);
      end
      t_end = cyc;
      go <= 1'b0;
      @(posedge clk);
      if (!cur.bar && !abd) begin
        chk <= 1'b1;
        @(posedge clk);
        chk <= 1'b0;
        @(posedge clk);
      end
    endtask
  end

  // ---- the links: card 0's port p to card 1's port p (CROSS: to its port 1 - p)
  for (genvar p = 0; p < 2; p++) begin : g_k
    localparam int Q = CROSS ? 1 - p : p;  // card 1's port
    if (LSIM) begin : g_sim
      otpu_link_sim #(.ID(p), .CA(0), .PA(p), .CB(1), .PB(Q), .NC(NCD)) u_link (
        .clk, .rst(lrst),
        .a_s_tdata(lo[0][p].s_tdata), .a_s_tvalid(lo[0][p].s_tvalid),
        .a_s_tready(li[0][p].s_tready), .a_s_tlast(lo[0][p].s_tlast),
        .a_m_tdata(li[0][p].m_tdata), .a_m_tvalid(li[0][p].m_tvalid),
        .a_m_tready(lo[0][p].m_tready), .a_m_tlast(li[0][p].m_tlast),
        .a_m_thdr(li[0][p].m_thdr), .a_up(li[0][p].up), .a_rx_err(li[0][p].rx_err),
        .a_abort_in(li[0][p].abort_in), .a_abort(lo[0][p].abort_req),
        .b_s_tdata(lo[1][Q].s_tdata), .b_s_tvalid(lo[1][Q].s_tvalid),
        .b_s_tready(li[1][Q].s_tready), .b_s_tlast(lo[1][Q].s_tlast),
        .b_m_tdata(li[1][Q].m_tdata), .b_m_tvalid(li[1][Q].m_tvalid),
        .b_m_tready(lo[1][Q].m_tready), .b_m_tlast(li[1][Q].m_tlast),
        .b_m_thdr(li[1][Q].m_thdr), .b_up(li[1][Q].up), .b_rx_err(li[1][Q].rx_err),
        .b_abort_in(li[1][Q].abort_in), .b_abort(lo[1][Q].abort_req),
        .a_stat(), .a_hello(), .b_stat(), .b_hello());
    end else begin : g_stub
      otpu_link_stub #(.ID(p)) u_link (
        .clk, .rst(lrst), .ao(lo[0][p]), .ai(li[0][p]), .bo(lo[1][Q]), .bi(li[1][Q]));
    end
  end

  // ---- the run: each card walks the list (card 1's run starts c1late cycles after card 0's)
  logic [NCD-1:0] idle = '0;
  bit relinked = 1'b0;
  bit relink = 1'b0, reset_links = 1'b0;
  bit     ready = 1'b0;                    // the list is made
  longint tstart, tend [NCD][$];
  for (genvar c = 0; c < NCD; c++) begin : g_w
    initial begin
      longint te;
      fin[c] = 1'b0;
      wait (ready);
      if (c == 1 && c1late > 0) begin
        repeat (c1late) @(posedge clk);
        crst[1] <= 1'b0;
        repeat (2 * XRS + 12) @(posedge clk);
      end
      for (int n = 0; n < ops.size(); n++) begin
        if (relink && n == ops.size() / 2) begin
          idle[c] = 1'b1;
          wait (relinked);
        end
        if (c == 1) repeat (c1wait) @(posedge clk);
        g_c[c].run_op(n, te);
        tend[c].push_back(te);
        repeat (rnd(0, gap) + (c == 1 ? rnd(0, cskew) : 0)) @(posedge clk);
      end
      fin[c] = 1'b1;
    end
  end

  initial begin
    reset_links = $test$plusargs("lrst_idle");
    relink = reset_links || $test$plusargs("run_idle");
    repeat (20) @(posedge clk);
    lrst <= 1'b0;
    crst <= c1late > 0 ? 2'b10 : 2'b00;
    repeat (2 * XRS + 24) @(posedge clk);  // (the links up: otpu_link_sim UPD 16)
    if (sizes) begin
      int sz [3] = '{16, 256, 4096};
      for (int i = 0; i < 3; i++)
        for (int k = 0; k < reps; k++) ops.push_back(mk_gather(1, sz[i]));
    end else if (big) begin
      for (int n = 0; n < nops; n++) ops.push_back(mk_gather(1, bigcols));
    end else begin
      for (int n = 0; n < nops; n++)
        if (int'($urandom_range(99)) < 20) ops.push_back(mk_bar());
        else ops.push_back(mk_gather(rnd(1, 6), rnd(1, 5 * L + 3)));
    end
    tstart = cyc;
    ready = 1'b1;
    if (relink) begin
      wait (idle == '1);
      repeat (2 * XRS + 40) @(posedge clk);
      crst <= '1;
      if (reset_links) lrst <= 1'b1;
      repeat (20) @(posedge clk);
      lrst <= 1'b0;
      repeat (2 * XRS + 40) @(posedge clk);
      crst <= '0;
      repeat (2 * XRS + 12) @(posedge clk);
      relinked = 1'b1;
    end
    wait (fin == '1);
    if (sizes) begin
      longint prev;
      prev = tstart;
      for (int i = 0; i < 3; i++) begin
        longint t1, tn, tb;
        for (int k = 0; k < reps; k++) begin
          int n;
          n = i * reps + k;
          tb = tend[0][n] > tend[1][n] ? tend[0][n] : tend[1][n];
          if (k == 0) t1 = tb - prev;
          tn = tb - prev;
          prev = tb;
        end
        $display("TBCX SIZE XRS=%0d words=%0d first=%0d last=%0d", XRS, sz_of(i), t1, tn);
      end
    end
    for (int c = 0; c < NCD; c++)
      $display("TBCX STATUS card=%0d xst=%02h seq=%0d drain=%0d", c,
               c == 0 ? g_c[0].xst : g_c[1].xst, c == 0 ? g_c[0].xseq : g_c[1].xseq,
               c == 0 ? g_c[0].ndrain : g_c[1].ndrain);
    $display("TBCX AHEAD cycles=%0d/%0d", g_c[0].nahead, g_c[1].nahead);
    $display("TBCX FOLLOW ops=%0d tagged=%0d/%0d", g_c[0].nfol, g_c[0].ntag, g_c[1].ntag);
    $display("TBCX DONE XRS=%0d ops=%0d gathers=%0d bars=%0d writes=%0d/%0d cycles=%0d",
             XRS, ops.size(), g_c[0].gathers, g_c[0].bars, g_c[0].nwr, g_c[1].nwr, cyc);
    $finish;
  end
  function automatic int sz_of(int i);
    return i == 0 ? 16 : i == 1 ? 256 : 4096;
  endfunction
endmodule
