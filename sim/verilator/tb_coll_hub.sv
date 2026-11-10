// The collective across the multi-slice board's crossings, alone: otpu_coll (RQ 0) behind
// otpu_coll_hub, S slices each XRS register stages away each way (otpu_xr, as otpu_board g_xs),
// each with its otpu_coll_port and a model of its TMEM's collective port (a granted read's words
// the next cycle, a granted write's words in memory at the cycle's end), the grant withheld at
// random (+stall=P percent of the cycles, in runs of up to +burst=B). Each slice's driver asks
// for the collective as otpu_slice does (coll_req held until its ack comes back; the slices start
// up to +skew=K cycles apart) for +ops=N collectives: GATHERs of random rows, cols and strides
// (each slice's source in the low half of its TMEM, the destination in the high half) and BARs,
// back to back (+gap=G: up to G idle cycles between them). Checked: every slice's TMEM writes,
// one at a time, in order, are the reference's (the unit's loops: slice, row, LANES words: their
// lanes, addresses and the source words), every GATHER's destination in every slice holds the
// sources' words afterwards, and no slice's collective is acknowledged before its writes are all
// in. +rst=P: a reset of the core (otpu_board core_rst: the hub and the unit a register later,
// each port through its crossing) cuts P percent of the GATHERs mid-flight, for 2 XRS + 4 .. 40
// cycles; once it is over every FIFO, credit and count is back at its reset value, and the
// collectives go on. +illegal: GATHER 3's destination overlaps slice 0's source (one LANES chunk
// on: a write over words reads still out read), which the hub's check stops. +sizes: instead of
// random ones, +reps=R GATHERs back to back of one row of each of 16, 256 and 4,096 words a slice,
// timed (TBCH SIZE: the cycles from the first request to the last slice's ack, and their slope).
// +seed=N.
module tb_coll_hub;
  import otpu_pkg::*;
  parameter int S     = 2;
  parameter int LANES = 8;
  parameter bit PIPE  = 1'b1;             // otpu_board COLL_PIPE
  parameter int XRS   = 2;                // otpu_board XRS
  parameter int TW    = 1 << 16;          // TMEM words a slice
  localparam int NR = PIPE ? 2 ** $clog2(2 * XRS + 8) : 1;     // (otpu_board CNR, CF)
  localparam int F  = PIPE ? 2 ** $clog2(2 * XRS + 6) : 8;
  localparam int L  = LANES;
  localparam int HW = TW / 2;             // the sources below, the destinations above

  logic clk = 1'b0;
  always #5 clk = ~clk;
  longint cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  int seed = 1, nops = 200, stall = 20, burst = 6, skew = 6, gap = 4, rstp = 0, reps = 4;
  bit illegal = 1'b0, sizes = 1'b0;
  initial begin
    void'($value$plusargs("seed=%d", seed));
    void'($value$plusargs("ops=%d", nops));
    void'($value$plusargs("stall=%d", stall));
    void'($value$plusargs("burst=%d", burst));
    void'($value$plusargs("skew=%d", skew));
    void'($value$plusargs("gap=%d", gap));
    void'($value$plusargs("rst=%d", rstp));
    void'($value$plusargs("reps=%d", reps));
    illegal = $test$plusargs("illegal");
    sizes = $test$plusargs("sizes");
    void'($urandom(seed));
  end

  // ---- the core's reset (otpu_board core_rst) and the collective's, a register later
  logic rst = 1'b1, coll_rst;
  always_ff @(posedge clk) coll_rst <= rst;

  // ---- the unit and the hub (otpu_board g_coll, g_xs u_chub)
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

  logic                        h_ack, h_wv;
  logic [S-1:0]                h_rv, h_req, h_dv, h_pop;
  logic [L-1:0]                h_rm, h_wm;
  logic [L-1:0][31:0]          h_ra, h_wa, h_wd;
  logic [S-1:0][L-1:0][31:0]   h_dd;
  cmd_t                        h_cmd [S];
  otpu_coll_hub #(.S(S), .LANES(L), .F(F), .NR(NR)) u_chub (
    .clk, .rst(coll_rst),
    .u_req(coll_req), .u_cmds(coll_cmds), .u_gnt(coll_gnt), .u_ack(coll_ack),
    .u_ren(coll_ren), .u_raddr(coll_raddr), .u_rdata(coll_rdata),
    .u_wen(coll_wen), .u_waddr(coll_waddr), .u_wdata(coll_wdata),
    .x_req(h_req), .x_cmds(h_cmd), .x_ack(h_ack), .x_rv(h_rv), .x_rm(h_rm), .x_ra(h_ra),
    .x_dv(h_dv), .x_dd(h_dd), .x_wv(h_wv), .x_wm(h_wm), .x_wa(h_wa), .x_wd(h_wd),
    .x_pop(h_pop));

  // ---- the run: each collective's command per slice, the reference's writes
  typedef struct packed {
    logic [L-1:0]       m;
    logic [L-1:0][31:0] a, d;
  } wr_t;
  logic [31:0] src0 [S][HW];               // each slice's sources (never written)
  initial for (int s = 0; s < S; s++) for (int i = 0; i < HW; i++) src0[s][i] = $urandom;
  cmd_t        cur [S];                    // the collective the drivers ask for now
  bit          go = 1'b0;                  // the drivers start it
  int          opn = 0;                    // its index
  logic [S-1:0] done;                      // each slice's ack came (g_s dn)
  bit          chk = 1'b0;                 // each slice checks the destination (g_s)
  wr_t         exl [$];                    // the collective's writes, in order (every slice's)
  bit          cut = 1'b0;                 // the collective was cut by a reset
  int          resets = 0, gathers = 0, bars = 0, words = 0;

  // ---- each slice: the crossing (XRS stages each way), its port, its TMEM's model, its driver
  logic [S-1:0] x_rst, sc_req, sc_ack, gl;
  cmd_t         sc_cmd [S];
  for (genvar s = 0; s < S; s++) begin : g_s
    localparam int DW = 1 + 32 * L + 32 * L + L + 32 * L + L + 3;
    localparam int UW = $bits(cmd_t) + 32 * L + 3;
    logic [DW-1:0] dn_d, dn_q;
    logic [UW-1:0] up_d, up_q;
    logic [XRS-2:0][DW-1:0] dn_m;
    logic [XRS-2:0][UW-1:0] up_m;
    logic               rx_crst, p_rv, p_wv, p_dv, p_pop;
    logic [L-1:0]       p_rm, p_wm, cren, cwen;
    logic [L-1:0][31:0] p_ra, p_wa, p_wd, p_dd, craddr, crdata, cwaddr, cwdata;
    assign dn_d = {rst, h_wd, h_wa, h_wm, h_ra, h_rm, h_wv, h_rv[s], h_ack};
    otpu_xr #(.W(DW), .RW(3)) u_dn_tx (.clk, .rst(coll_rst), .d(dn_d), .q(dn_m[0]));
    for (genvar i = 1; i < XRS - 1; i++) begin : g_dm
      otpu_xr #(.W(DW), .RW(3)) u_dn (.clk, .rst(coll_rst), .d(dn_m[i-1]), .q(dn_m[i]));
    end
    otpu_xr #(.W(DW), .RW(3)) u_dn_rx (.clk, .rst(rx_crst), .d(dn_m[XRS-2]), .q(dn_q));
    assign {rx_crst, p_wd, p_wa, p_wm, p_ra, p_rm, p_wv, p_rv, sc_ack[s]} = dn_q;
    assign x_rst[s] = rx_crst;

    otpu_coll_port #(.LANES(L), .F(F), .NR(NR)) u_cport (
      .clk, .rst(x_rst[s]), .rv(p_rv), .rm(p_rm), .ra(p_ra), .wv(p_wv), .wm(p_wm), .wa(p_wa),
      .wd(p_wd), .dv(p_dv), .dd(p_dd), .pop(p_pop),
      .coll_ren(cren), .coll_raddr(craddr), .coll_rdata(crdata),
      .coll_wen(cwen), .coll_waddr(cwaddr), .coll_wdata(cwdata),
      .gnt_local(gl[s]), .gnt());

    assign up_d = {sc_cmd[s], p_dd, p_pop, p_dv, sc_req[s]};
    otpu_xr #(.W(UW), .RW(3)) u_up_tx (.clk, .rst(x_rst[s]), .d(up_d), .q(up_m[0]));
    for (genvar i = 1; i < XRS - 1; i++) begin : g_um
      otpu_xr #(.W(UW), .RW(3)) u_up (.clk, .rst(x_rst[s]), .d(up_m[i-1]), .q(up_m[i]));
    end
    otpu_xr #(.W(UW), .RW(3)) u_up_rx (.clk, .rst(coll_rst), .d(up_m[XRS-2]), .q(up_q));
    assign {h_cmd[s], h_dd[s], h_pop[s], h_dv[s], h_req[s]} = up_q;

    // the TMEM's collective port: the grant withheld in random runs
    logic [31:0] mem [TW];
    logic [L-1:0][31:0] rq;
    int run_left = 0;
    int ix = 0;                            // its next write's in exl
    int nw = 0;                            // writes seen
    initial for (int i = 0; i < TW; i++) mem[i] = (i < HW) ? src0[s][i] : 32'hDEAD0000 | i;
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
        w.m = cwen;
        for (int l = 0; l < L; l++) begin
          w.a[l] = cwen[l] ? cwaddr[l] : '0;
          w.d[l] = cwen[l] ? cwdata[l] : '0;
          if (cwen[l]) mem[cwaddr[l] % TW] <= cwdata[l];
        end
        if (ix >= exl.size())
          $fatal(1, "tb_coll_hub: slice %0d wrote %0d (%h) with no write expected (op %0d)", s,
                 w.a[0], w.m, opn);
        e = exl[ix];
        if (w != e)
          $fatal(1, "tb_coll_hub: slice %0d write %0d of op %0d: lanes %h addr %0d data %h, expected lanes %h addr %0d data %h",
                 s, ix, opn, w.m, w.a[0], w.d[0], e.m, e.a[0], e.d[0]);
        ix <= ix + 1;
        nw <= nw + 1;
      end
      if (!go) ix <= 0;
    end

    // the driver: coll_req from the start until its ack, as otpu_slice's (reset by its core's)
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
          sc_req[s] <= 1'b1;
          sc_cmd[s] <= cur[s];
          asked <= 1'b1;
        end else wait_left <= wait_left - 1;
      end else if (sc_req[s] && sc_ack[s]) begin
        sc_req[s] <= 1'b0;
        if (ix != exl.size())
          $fatal(1, "tb_coll_hub: slice %0d's ack of op %0d with %0d writes still to come", s, opn,
                 exl.size() - ix);
        dn <= 1'b1;
      end
    end
    initial sc_req[s] = 1'b0;

    // the destination after a GATHER (chk): the sources' words
    always @(posedge clk)
      if (chk) begin
        int rows, cols;
        rows = int'(cur[0].w3[15:0]); cols = int'(cur[0].w3[31:16]);
        for (int k = 0; k < S; k++)
          for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) begin
              int da, sa;
              da = int'(cur[0].w2) + k * int'(cur[0].w6) + r * int'(cur[0].w5) + c;
              sa = int'(cur[k].w1) + r * int'(cur[k].w4) + c;
              if (mem[da % TW] != src0[k][sa % HW])
                $fatal(1, "tb_coll_hub: slice %0d's word %0d (slice %0d row %0d col %0d) after op %0d: %h, not %h",
                       s, da, k, r, c, opn, mem[da % TW], src0[k][sa % HW]);
            end
      end
  end

  // ---- the hub's and the ports' state at its reset values (after a reset)
  if (PIPE) begin : g_cc
    task automatic check_clean();
      if (u_chub.g_pipe.nout != 0 || u_chub.g_pipe.on != 0 || u_chub.g_pipe.wn != 0 ||
          u_chub.g_pipe.rn != '0 || u_chub.g_pipe.wo_v || !u_chub.g_pipe.g_ok ||
          u_chub.o != '0 || u_chub.pend || u_chub.blk != '0)
        $fatal(1, "tb_coll_hub: the hub not clean after the reset (out %0d, order %0d, writes %0d, o %h)",
               u_chub.g_pipe.nout, u_chub.g_pipe.on, u_chub.g_pipe.wn, u_chub.o);
    endtask
  end else begin : g_cc
    task automatic check_clean();
      if (u_chub.g_one.rd_wait || u_chub.o != '0 || u_chub.pend || u_chub.blk != '0)
        $fatal(1, "tb_coll_hub: the hub not clean after the reset");
    endtask
  end
  for (genvar s = 0; s < S; s++) begin : g_ck
    logic qn0;
    if (PIPE) begin : g_q
      assign qn0 = g_s[s].u_cport.g_pipe.qn == 0;
    end else begin : g_q
      assign qn0 = 1'b1;
    end
    always @(posedge clk)
      if (!rst && !x_rst[s] && cut && !go && sc_req == '0) begin
        if (g_s[s].u_cport.n != 0 || g_s[s].u_cport.hv || g_s[s].u_cport.dv || g_s[s].u_cport.pop)
          $fatal(1, "tb_coll_hub: slice %0d's port not clean after the reset", s);
        if (!qn0)
          $fatal(1, "tb_coll_hub: slice %0d's port read FIFO not clean after the reset", s);
      end
  end

  // ---- the run
  function automatic int rnd(int lo, int hi);
    return lo + int'($urandom_range(hi - lo));
  endfunction
  // a GATHER (rows x cols a slice); the reference's writes into every slice's queue
  task automatic gather(int rows, int cols, bit bad);
    logic [31:0] w1 [S], w4 [S];
    int drs, seg, dst, span;
    exl.delete();
    for (int s = 0; s < S; s++) begin
      w4[s] = 32'(cols + rnd(0, 8) * (rows > 1));
      w1[s] = 32'(rnd(0, HW - ((rows - 1) * int'(w4[s]) + cols)));
    end
    drs = cols + rnd(0, 8) * (rows > 1);
    seg = (rows - 1) * drs + cols + rnd(0, 16);
    span = (S - 1) * seg + (rows - 1) * drs + cols;
    dst = HW + rnd(0, HW - span);
    if (bad) begin                         // slice 0's source, a chunk on: overlaps
      w1[0] = 32'(HW - (rows - 1) * int'(w4[0]) - cols - LANES);
      dst = int'(w1[0]) + LANES;
    end
    for (int s = 0; s < S; s++) begin
      cur[s] = '0;
      cur[s].op = OP_GATHER;
      cur[s].w1 = w1[s];
      cur[s].w2 = 32'(dst);
      cur[s].w3 = {16'(cols), 16'(rows)};
      cur[s].w4 = w4[s];
      cur[s].w5 = 32'(drs);
      cur[s].w6 = 32'(seg);
    end
    for (int k = 0; k < S; k++)
      for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c += L) begin
          wr_t e;
          e = '0;
          for (int l = 0; l < L; l++)
            if (c + l < cols) begin
              int sa;
              e.m[l] = 1'b1;
              e.a[l] = 32'(dst + k * seg + r * drs + c + l);
              sa = int'(w1[k]) + r * int'(w4[k]) + c + l;
              e.d[l] = (sa < HW) ? src0[k][sa] : 32'hDEAD0000 | sa;
            end
          exl.push_back(e);
        end
    gathers++;
    words += rows * cols;
  endtask
  task automatic bar();
    exl.delete();
    for (int s = 0; s < S; s++) begin cur[s] = '0; cur[s].op = OP_BAR; end
    bars++;
  endtask
  // every slice's destination holds the sources' words (cur: the GATHER just done; g_s)
  task automatic check_dst();
    chk <= 1'b1;
    @(posedge clk);
    chk <= 1'b0;
    @(posedge clk);
  endtask
  task automatic run_one(output longint took);
    longint t0;
    int rlen, rat;
    bit rcut;
    cut = 1'b0;
    rcut = rstp > 0 && cur[0].op == OP_GATHER && int'($urandom_range(99)) < rstp;
    rat = rnd(4, 60);
    rlen = rnd(2 * XRS + 4, 40);
    @(posedge clk);
    go <= 1'b1;
    t0 = cyc;
    while (done != '1) begin
      @(posedge clk);
      if (rcut && cyc - t0 == longint'(rat)) begin
        rst <= 1'b1;                       // cut it: the core's reset
        repeat (rlen) @(posedge clk);
        rst <= 1'b0;
        resets++;
        cut = 1'b1;
        go <= 1'b0;
        repeat (3 * XRS + 8) @(posedge clk);
        g_cc.check_clean();
        break;
      end
      if (cyc - t0 > 2000000) $fatal(1, "tb_coll_hub: op %0d hangs (done %b)", opn, done);
    end
    took = cyc - t0;
    go <= 1'b0;
    @(posedge clk);
  endtask

  initial begin
    longint took;
    repeat (20) @(posedge clk);
    rst <= 1'b0;
    repeat (2 * XRS + 4) @(posedge clk);
    if (sizes) begin
      int sz [3] = '{16, 256, 4096};
      for (int i = 0; i < 3; i++) begin
        longint t1, tn;
        t1 = 0;
        for (int k = 0; k < reps; k++) begin
          gather(1, sz[i], 1'b0);
          run_one(took);
          check_dst();
          if (k == 0) t1 = took;
          tn = took;
          opn++;
        end
        $display("TBCH SIZE S=%0d PIPE=%0d XRS=%0d words=%0d first=%0d last=%0d", S, PIPE, XRS,
                 sz[i], t1, tn);
      end
      // back to back: reps GATHERs of 256 words, timed together
      begin
        longint t0b;
        t0b = cyc;
        for (int k = 0; k < reps; k++) begin
          gather(1, 256, 1'b0);
          run_one(took);
          opn++;
        end
        $display("TBCH B2B S=%0d PIPE=%0d XRS=%0d words=256 n=%0d cycles=%0d", S, PIPE, XRS, reps,
                 cyc - t0b);
      end
    end else begin
      for (opn = 0; opn < nops; opn++) begin
        if (illegal && opn == 3) gather(rnd(1, 3), rnd(3 * L, 6 * L), 1'b1);
        else if (int'($urandom_range(99)) < 20) bar();
        else gather(rnd(1, 6), rnd(1, 5 * L + 3), 1'b0);
        run_one(took);
        if (cur[0].op == OP_GATHER && !cut) check_dst();
        repeat (rnd(0, gap)) @(posedge clk);
      end
    end
    $display("TBCH DONE S=%0d PIPE=%0d XRS=%0d NR=%0d F=%0d ops=%0d gathers=%0d bars=%0d words=%0d writes=%0d resets=%0d cycles=%0d",
             S, PIPE, XRS, NR, F, opn, gathers, bars, words, g_s[0].nw, resets, cyc);
    $finish;
  end
endmodule
