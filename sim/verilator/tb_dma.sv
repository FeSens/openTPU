// Unit test of rtl/dma/otpu_dma.sv's LD, ST and WAITW (tests/test_dma_ldq.py), with LDA its LDA
// too: random command sequences issued as the sequencer issues them (start a register, set on rdy
// and never two cycles running), against a model of DRAM port B, of TMEM and (LDA) of ACT RAM.
//   Port B grants at random (gnt_pct). A read returns in order, lat_min .. lat_max cycles after
//   its grant, with the chunk as it was at the grant. A write is posted: it lands in DRAM at its
//   acknowledgement, wlat_min .. wlat_max cycles after its grant, and wr_idle is low while one is
//   pending. The writes go on +nch=N channels (default 1; chunk c on channel (c ^ c >> 3) % N),
//   acknowledged in order on each and independently across them: a later write on one channel may
//   land before an earlier one on another (as otpu_native_dram's channels, whose n_wdone counts
//   are all the DMA's wr_idle sees).
//   TMEM: a write lands at the edge it is presented; a read's word comes the next cycle and holds
//   until the lane reads again.
//   ACT RAM (LDA): AROWS rows of ABLK blocks of D bytes and ABLK 8-byte records ({d, S}); a byte
//   or record write lands at the edge it is presented; act_busy (the quantizer's write) is high
//   act_busy_pct % of the cycles (+act_busy=).
// Checked as it runs ($fatal), sq being the started commands not yet done, in order:
//   - each done is sq's head's; an ST's comes with none of its own writes pending; err never
//     rises; no more than LDQ commands are in flight (sq, after the cycle's done, as a start is
//     presented);
//   - TMEM writes go to the head's words only, TMEM reads come from an ST head's words, or with
//     LDQ > 1 from an ST's behind an ST head whose chunks have all been written (`early_ss`);
//   - ACT writes go to an LDA head's rows and blocks only, never beside act_busy;
//   - DRAM writes go to an ST head's chunks only (an ST behind an ST writes once the one before is
//     done: its writes are held, so that the done, at wr_idle, is that ST's own writes'); a DRAM
//     read is for the head's chunks (an LDA's:
//     its rows'), or with LDQ > 1 the next command's (an LD or LDA started under the head:
//     counted, `early`; `early_la` those with an LDA in the head or the next);
//   - no read is granted while a write it would see stale is pending (a word of the chunk that
//     the reading command uses, in a write not yet acknowledged).
// (otpu_dma itself checks its chunk buffer: no more than DEPTH chunks reserved; with LDQ 2 the
// front's requests count the back's unreturned ones.)
// Input (+in=, hex): "n gnt_pct lat_min lat_max wlat_min wlat_max start_pct seed max_cycles", then
// n lines "op flags a1 w2 w3 w4 w5 w6" (a1 the DRAM byte address, AW bits). +dram= / +tmem=: the
// initial words ($readmemh; NDW and NTW words, DRAM word addresses taken mod NDW).
// Output (+out=): per done, "k cycle v0 .. v(m-1)": command k's destination words as the done
// rises (an LD's m TMEM words, an ST's DRAM words, a WAITW's TMEM word, an LDA's rows: each row's
// KB blocks' D bytes then their KB records, as 32-bit words); then "E cycles early early_st
// early_la early_ss wro" (early_st: the early reads under an ST head, in its write wait; early_ss
// the cycles an ST behind an ST read TMEM before the head's done; wro the writes that landed
// before one granted earlier: the channels' reordering).
// +dram_out= / +tmem_out=: the final words.
module tb_dma;
  import otpu_pkg::*;
  parameter int D         = 128;
  parameter int LANES     = 8;
  parameter int DEPTH     = 512;
  parameter int HAS_DSTEP = 1;
  parameter int AW        = 36;
  parameter int LDQ       = 1;
  parameter int NDW       = 1 << 15;
  parameter int NTW       = 1 << 14;
  parameter bit LDA       = 1'b0;
  parameter int ALANES    = 16;
  parameter int AROWS     = 16;
  parameter int ABLK      = 128;
  localparam int WA = waw(AW);
  localparam int CW = D / 4;

  logic clk = 1'b0, rst = 1'b1;
  always #5 clk = !clk;

  logic                   start = 1'b0;
  cmd_t                   cmd;
  logic                   rdy, done, err, ww_held;
  logic                   b_req, b_gnt = 1'b0, b_we, b_rvalid = 1'b0, wr_idle = 1'b1;
  logic [CW-1:0]          b_wmask;
  logic [D*8-1:0]         b_wdata, b_rdata;
  logic [WA-1:0]          b_addr;
  logic [LANES-1:0]       t_ren, t_rben, t_wen;
  logic [LANES-1:0][31:0] t_raddr, t_rbad, t_rdata, t_waddr, t_wdata;
  logic                   ss_req, ss_pe, ss_in_v;
  ss_cfg_t                ss_cfg;
  logic [31:0]            ss_in_d [LANES], ss_fd [LANES], ss_y_d [LANES];
  logic [2:0]             ss_fk;
  logic [4:0]             ss_fi;
  for (genvar l = 0; l < LANES; l++) begin : g_y
    assign ss_y_d[l] = '0;
  end
  logic                    act_busy = 1'b0, asc_we;
  logic [ALANES-1:0]       act_we;
  logic [7:0]              act_row, asc_row;
  logic [31:0]             act_idx;
  logic [ALANES-1:0][7:0]  act_data;
  logic [15:0]             asc_blk;
  logic [63:0]             asc_data;

  otpu_dma #(.D(D), .LANES(LANES), .DEPTH(DEPTH), .LDQ(LDQ), .HAS_DSTEP(HAS_DSTEP), .AW(AW),
             .LDA(LDA), .ALANES(ALANES)) dut (
    .clk, .rst, .start, .cmd, .rdy, .done, .err, .ww_held,
    .b_req, .b_gnt, .b_we, .b_wmask, .b_wdata, .b_addr, .b_rvalid, .b_rdata, .wr_idle,
    .t_ren, .t_raddr, .t_rben, .t_rbad, .t_rdata, .t_wen, .t_waddr, .t_wdata,
    .ss_req, .ss_gnt(1'b0), .ss_cfg, .ss_pe, .ss_in_v, .ss_in_d, .ss_fk, .ss_fi, .ss_fd,
    .ss_y_v(1'b0), .ss_y_d, .ss_o_v(1'b0), .ss_o_d('0),
    .act_busy, .act_we, .act_row, .act_idx, .act_data, .asc_we, .asc_row, .asc_blk, .asc_data);

  logic [31:0] dram [NDW];
  logic [31:0] tmem [NTW];
  logic [7:0]  actb [AROWS][ABLK * D];
  logic [63:0] acts [AROWS][ABLK];
  int unsigned act_busy_pct = 0;

  // the commands, and each one's DRAM words [c_da, c_da + c_dn) and TMEM words [c_w2, c_w2 + c_tn)
  int unsigned n, gnt_pct, lat_min, lat_max, wlat_min, wlat_max, start_pct, seed;
  longint      max_cyc;
  int unsigned c_op [], c_fl [], c_w2 [], c_w3 [], c_w4 [], c_w5 [], c_w6 [], c_tn [];
  longint      c_a1 [], c_da [], c_dn [];
  int          sq [$];
  int          nxt = 0;
  longint      cyc = 0, early = 0, early_st = 0, early_la = 0;   // early: under an ST's write wait too
  longint      early_ss = 0, wro = 0;   // wro: writes landed before an earlier-granted one
  int          wgnt [];                 // each command's chunk writes granted
  int unsigned nch = 1;

  typedef struct packed { longint t; logic [D*8-1:0] d; } rd_t;
  typedef struct packed { longint t; logic [63:0] a; logic [CW-1:0] m; logic [D*8-1:0] d; int k; } wr_t;
  rd_t rq [$];
  wr_t wq [$];                          // every channel's (each one's in order of t)
  longint rlast = 0;
  longint wlast [16];
  function automatic int chan_of(input longint ch);   // chunk address ch's channel
    longint c = ch / CW;
    return int'((c ^ (c >> 3)) % nch);
  endfunction
  function automatic int nck(input int k);            // ST k's chunks
    return c_dn[k] == 0 ? 0 : int'((c_da[k] + c_dn[k] - 1) / CW - c_da[k] / CW + 1);
  endfunction

  // an LDA's rows (w2: rows, ab << 8, KB << 16; w3 the row stride in bytes; w4 the first ACT row):
  // row r's DRAM words [c_da + r * w3 / 4, ... + la_rw(k))
  function automatic longint la_rw(input int k);
    return longint'(c_w2[k] >> 16) * (D / 4 + 2);
  endfunction
  function automatic logic in_cmd(input int k, input longint a);  // DRAM word a is k's
    if (c_op[k] != OP_LDA) return a >= c_da[k] && a < c_da[k] + c_dn[k];
    for (int r = 0; r < int'(c_w2[k] & 8'hFF); r++)
      if (a >= c_da[k] + longint'(r) * (c_w3[k] / 4) &&
          a < c_da[k] + longint'(r) * (c_w3[k] / 4) + la_rw(k)) return 1'b1;
    return 1'b0;
  endfunction
  function automatic logic [CW-1:0] words_of(input int k, input longint ch);
    logic [CW-1:0] m;
    for (int i = 0; i < CW; i++) m[i] = in_cmd(k, ch + i);
    return m;
  endfunction
  function automatic logic hits(input int k, input longint ch);   // chunk ch has a word of k's
    return k >= 0 && c_dn[k] != 0 && words_of(k, ch) != '0;
  endfunction
  function automatic logic t_in(input int k, input longint a);    // TMEM word a is k's
    return a >= longint'(c_w2[k]) && a < longint'(c_w2[k]) + longint'(c_tn[k]);
  endfunction

  int fout;
  always @(posedge clk) if (!rst) begin
    int h, h1, o;
    longint ch, lt;
    logic [D*8-1:0] d;
    logic [CW-1:0]  m;
    rd_t r;
    wr_t w;
    cyc = cyc + 1;
    if (err) $fatal(1, "cycle %0d: err (a WAITW timed out)", cyc);
    // the done: the head's destination words as they stand (this cycle's writes land after)
    if (done) begin
      if (sq.size() == 0) $fatal(1, "cycle %0d: done with no command outstanding", cyc);
      h = sq.pop_front();
      if (c_op[h] == OP_ST)
        foreach (wq[i])
          if (wq[i].k == h)
            $fatal(1, "cycle %0d: ST %0d done with its write of %h pending", cyc, h, wq[i].a);
      $fwrite(fout, "%0d %0d", h, cyc);
      if (c_op[h] == OP_ST)
        for (longint i = 0; i < c_dn[h]; i++) $fwrite(fout, " %h", dram[(c_da[h] + i) % NDW]);
      else if (c_op[h] == OP_LDA)
        for (int r = 0; r < int'(c_w2[h] & 8'hFF); r++) begin
          for (int i = (c_w2[h] >> 8 & 8'hFF) * D; i < ((c_w2[h] >> 8 & 8'hFF) + (c_w2[h] >> 16)) * D;
               i += 4)
            $fwrite(fout, " %h", {actb[c_w4[h] + r][i + 3], actb[c_w4[h] + r][i + 2],
                                  actb[c_w4[h] + r][i + 1], actb[c_w4[h] + r][i]});
          for (int b = c_w2[h] >> 8 & 8'hFF; b < (c_w2[h] >> 8 & 8'hFF) + (c_w2[h] >> 16); b++)
            $fwrite(fout, " %h %h", acts[c_w4[h] + r][b][31:0], acts[c_w4[h] + r][b][63:32]);
        end
      else
        for (int i = 0; i < int'(c_tn[h]); i++) $fwrite(fout, " %h", tmem[(c_w2[h] + i) % NTW]);
      $fwrite(fout, "\n");
    end
    if (start && sq.size() > int'(LDQ))
      $fatal(1, "cycle %0d: a start with %0d commands in flight (LDQ %0d)", cyc, sq.size(), LDQ);
    h = sq.size() > 0 ? sq[0] : -1;
    h1 = sq.size() > 1 ? sq[1] : -1;
    // TMEM: this cycle's reads (an ST's: the head's, or the next ST's once the head's chunks are
    // all written), then its writes (the head's)
    begin
      logic ss;
      ss = 1'b0;
      for (int l = 0; l < LANES; l++)
        if (t_ren[l]) begin
          if (h >= 0 && c_op[h] == OP_ST && t_in(h, longint'(t_raddr[l]))) ;
          else if (LDQ > 1 && h1 >= 0 && c_op[h] == OP_ST && c_op[h1] == OP_ST &&
                   wgnt[h] == nck(h) && t_in(h1, longint'(t_raddr[l])))
            ss = 1'b1;
          else
            $fatal(1, "cycle %0d: TMEM read of %0d, not the head's (command %0d, %0d of %0d chunks written; next %0d)",
                   cyc, t_raddr[l], h, h < 0 ? 0 : wgnt[h], h < 0 ? 0 : nck(h), h1);
          t_rdata[l] <= tmem[t_raddr[l] % NTW];
        end
      if (ss) early_ss = early_ss + 1;
    end
    for (int l = 0; l < LANES; l++)
      if (t_wen[l]) begin
        if (h < 0 || c_op[h] == OP_ST || c_op[h] == OP_LDA || !t_in(h, longint'(t_waddr[l])))
          $fatal(1, "cycle %0d: TMEM write of %0d, not the head's (command %0d)", cyc, t_waddr[l], h);
        tmem[t_waddr[l] % NTW] = t_wdata[l];
      end
    // ACT RAM: an LDA head's rows w4 .. and blocks ab .. ab + KB - 1
    if (act_we != '0 || asc_we) begin
      int ab, kb;
      if (h < 0 || c_op[h] != OP_LDA) $fatal(1, "cycle %0d: ACT write, head %0d not an LDA", cyc, h);
      if (act_busy) $fatal(1, "cycle %0d: ACT write beside the quantizer's", cyc);
      ab = int'(c_w2[h] >> 8 & 8'hFF); kb = int'(c_w2[h] >> 16);
      if (act_row < c_w4[h] || act_row >= c_w4[h] + (c_w2[h] & 8'hFF) || asc_row != act_row)
        $fatal(1, "cycle %0d: ACT write of row %0d, not LDA %0d's", cyc, act_row, h);
      if (act_we != '0 && (act_idx < ab * D || act_idx + ALANES > (ab + kb) * D))
        $fatal(1, "cycle %0d: ACT write at byte %0d, not LDA %0d's blocks", cyc, act_idx, h);
      if (asc_we && (asc_blk < ab || asc_blk >= ab + kb))
        $fatal(1, "cycle %0d: ACT record of block %0d, not LDA %0d's", cyc, asc_blk, h);
      for (int l = 0; l < ALANES; l++) if (act_we[l]) actb[act_row][act_idx + l] = act_data[l];
      if (asc_we) acts[asc_row][asc_blk] = asc_data;
    end
    act_busy <= ($urandom % 100) < act_busy_pct;
    // port B: acknowledged writes land (each channel's in order), then this cycle's request,
    // then next cycle's return
    for (int i = 0; i < wq.size(); )
      if (wq[i].t <= cyc) begin
        w = wq[i];
        for (int j = 0; j < CW; j++)
          if (w.m[j]) dram[(w.a + j) % NDW] = w.d[32 * j +: 32];
        if (i != 0) wro = wro + 1;
        wq.delete(i);
      end else
        i++;
    if (b_req && b_gnt) begin
      ch = longint'(b_addr);
      if (ch % CW != 0) $fatal(1, "cycle %0d: port B address %h not chunk aligned", cyc, ch);
      if (b_we) begin
        if (h < 0 || c_op[h] != OP_ST || !hits(h, ch) || (b_wmask & ~words_of(h, ch)) != 0)
          $fatal(1, "cycle %0d: DRAM write of %h mask %h, not the head's (command %0d)", cyc, ch,
                 b_wmask, h);
        lt = cyc + wlat_min + $urandom % (wlat_max - wlat_min + 1);
        o = chan_of(ch);
        w.t = lt > wlast[o] ? lt : wlast[o] + 1;
        w.a = 64'(ch); w.m = b_wmask; w.d = b_wdata; w.k = h;
        wlast[o] = w.t;
        wq.push_back(w);
        wgnt[h] = wgnt[h] + 1;
      end else begin
        o = hits(h, ch) && c_op[h] != OP_ST ? h : -1;
        if (o < 0 && LDQ > 1 && h1 >= 0 && (c_op[h1] == OP_LD || c_op[h1] == OP_LDA) &&
            hits(h1, ch)) begin
          o = h1;
          early = early + 1;
          if (c_op[h] == OP_ST) early_st = early_st + 1;
          if (c_op[h] == OP_LDA || c_op[h1] == OP_LDA) early_la = early_la + 1;
        end
        if (o < 0) $fatal(1, "cycle %0d: DRAM read of %h for neither command %0d nor %0d", cyc, ch,
                          h, h1);
        m = words_of(o, ch);
        foreach (wq[i])
          if (wq[i].a == 64'(ch) && (wq[i].m & m) != 0)
            $fatal(1, "cycle %0d: DRAM read of %h (command %0d) passes a pending write (mask %h)",
                   cyc, ch, o, wq[i].m);
        for (int i = 0; i < CW; i++) d[32 * i +: 32] = dram[(ch + i) % NDW];
        lt = cyc + lat_min + $urandom % (lat_max - lat_min + 1);
        r.t = lt > rlast ? lt : rlast + 1;
        r.d = d;
        rlast = r.t;
        rq.push_back(r);
      end
    end
    if (rq.size() > 0 && rq[0].t <= cyc + 1) begin
      r = rq.pop_front();
      b_rvalid <= 1'b1;
      b_rdata <= r.d;
    end else
      b_rvalid <= 1'b0;
    wr_idle <= wq.size() == 0;
    b_gnt <= ($urandom % 100) < gnt_pct;
    // the next command
    if (start) start <= 1'b0;
    else if (rdy && nxt < int'(n) && ($urandom % 100) < start_pct) begin
      cmd <= '0;
      cmd.op <= 8'(c_op[nxt]);
      cmd.flags <= 8'(c_fl[nxt]);
      cmd.w1 <= c_a1[nxt][31:0];
      cmd.w1h <= AWX'(c_a1[nxt] >> 32);
      cmd.w2 <= c_w2[nxt]; cmd.w3 <= c_w3[nxt]; cmd.w4 <= c_w4[nxt];
      cmd.w5 <= c_w5[nxt]; cmd.w6 <= c_w6[nxt];
      start <= 1'b1;
      sq.push_back(nxt);
      nxt <= nxt + 1;
    end
    if (cyc > max_cyc) $fatal(1, "timeout: %0d of %0d commands started, %0d outstanding", nxt, n,
                              sq.size());
  end

  string fname;
  int fin, r;
  initial begin
    if (!$value$plusargs("in=%s", fname)) $fatal(1, "need +in=");
    fin = $fopen(fname, "r");
    if (fin == 0) $fatal(1, "cannot open %s", fname);
    r = $fscanf(fin, "%h %h %h %h %h %h %h %h %h", n, gnt_pct, lat_min, lat_max, wlat_min, wlat_max,
                start_pct, seed, max_cyc);
    if (r != 9) $fatal(1, "bad header");
    void'($urandom(seed));
    c_op = new[n]; c_fl = new[n]; c_a1 = new[n]; c_w2 = new[n]; c_w3 = new[n]; c_w4 = new[n];
    c_w5 = new[n]; c_w6 = new[n]; c_tn = new[n]; c_da = new[n]; c_dn = new[n]; wgnt = new[n];
    foreach (wgnt[i]) wgnt[i] = 0;
    foreach (wlast[c]) wlast[c] = 0;
    void'($value$plusargs("nch=%d", nch));
    if (nch < 1 || nch > 16) $fatal(1, "+nch=%0d: 1 .. 16", nch);
    for (int i = 0; i < int'(n); i++) begin
      r = $fscanf(fin, "%h %h %h %h %h %h %h %h", c_op[i], c_fl[i], c_a1[i], c_w2[i], c_w3[i],
                  c_w4[i], c_w5[i], c_w6[i]);
      if (r != 8) $fatal(1, "bad command %0d", i);
      c_da[i] = (c_a1[i] & ((64'd1 << AW) - 1)) >> 2;
      c_dn[i] = c_op[i] == OP_WAITW ? 1 : longint'(c_w3[i]);
      c_tn[i] = c_op[i] == OP_WAITW ? 1 : c_w3[i];
      if (c_op[i] == OP_LDA) begin                 // (c_dn: its span, rows w3 apart)
        if (!LDA) $fatal(1, "command %0d: LDA on a tb without LDA", i);
        c_dn[i] = (c_w2[i] & 8'hFF) == 0 ? 0 :
                  longint'((c_w2[i] & 8'hFF) - 1) * (c_w3[i] / 4) + la_rw(i);
        c_tn[i] = 0;
      end
    end
    if (!$value$plusargs("dram=%s", fname)) $fatal(1, "need +dram=");
    $readmemh(fname, dram);
    if (!$value$plusargs("tmem=%s", fname)) $fatal(1, "need +tmem=");
    $readmemh(fname, tmem);
    void'($value$plusargs("act_busy=%d", act_busy_pct));
    foreach (actb[r, i]) actb[r][i] = '0;
    foreach (acts[r, b]) acts[r][b] = '0;
    if (!$value$plusargs("out=%s", fname)) $fatal(1, "need +out=");
    fout = $fopen(fname, "w");
    if (fout == 0) $fatal(1, "cannot open %s", fname);
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst = 1'b0;
    while (nxt < int'(n) || sq.size() != 0 || wq.size() != 0 || rq.size() != 0 || !rdy)
      @(posedge clk);
    repeat (4) @(posedge clk);
    if (b_req) $fatal(1, "a port B request after the last done");
    $fdisplay(fout, "E %0d %0d %0d %0d %0d %0d", cyc, early, early_st, early_la, early_ss, wro);
    $fclose(fout);
    if ($value$plusargs("dram_out=%s", fname)) $writememh(fname, dram);
    if ($value$plusargs("tmem_out=%s", fname)) $writememh(fname, tmem);
    $finish;
  end
endmodule
