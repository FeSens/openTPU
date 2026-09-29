// Unit test of the stream engine's front (rtl/vpu/otpu_vpu.sv with HAS_SE, tests/test_se_vpu.py):
// the VPU with its tail, driven like the DMA drives it (ss_req, the grant, the fills, the stream
// with random pe stalls, ss_req dropped after the last output), with independent VOPs running
// on a TMEM model around the streams: some are queued or in flight when a stream asks, so SE
// must drain them before the grant, and they resume after it. The VOPs share slot 0's partial
// loop and u_vt with the streams (RSUM/RSSQ/RDOT).
// Input (+in=, hex words): the VOPs, "nvop" then nvop lines "op flags w1 .. w7"; then per stream
//   1 ns rows a_en a_sel dmode g_src q_en nfill nseg pe_pct nrel delay pad64
// (nrel: VOPs released so far; delay: cycles from their release to ss_req), nfill lines
// "fk fi d0 .. d7" and nseg lines "d0 .. d7"; a 0 ends the file. +tmem= preloads the TMEM
// (readmemh). Output (+out=): per stream "Y d0 .. d7" per updated segment and "O d" per row
// output, then "E"; at the end "T" and the TMEM words (+dump= of them), one per line.
module tb_se_vpu #(parameter bit WBUF = 1'b1, parameter bit ONE_TREE = 1'b0,
                   parameter bit COMP8 = 1'b0,
                   parameter int GNT_PCT = 80);
  import otpu_pkg::*;
  import otpu_fp::*;
  localparam int L = 8;
  localparam int MAXSEG = 256 * 32;
  localparam int MAXV = 1024;
  localparam int TW = 1 << 16;

  logic clk = 1'b0, rst = 1'b1;
  always #5 clk = !clk;

  // the VPU's command port
  logic start = 1'b0, rdy, done, gnt = 1'b1, ren;
  cmd_t cmd;
  logic [L-1:0] ta_en, tb_en, tw_en;
  logic [L-1:0][31:0] ta_addr, tb_addr, ta_data, tb_data, tw_addr, tw_data;
  logic pf_u;
  logic [31:0] pf_frz;
  // the stream port
  logic ss_req = 1'b0, ss_gnt, ss_pe = 1'b0, ss_in_v = 1'b0, ss_y_v, ss_o_v;
  ss_cfg_t ss_cfg;
  f32_t ss_in_d [L], ss_fd [L], ss_y_d [L];
  f32_t ss_o_d;
  logic [2:0] ss_fk = SF_NONE;
  logic [4:0] ss_fi = '0;

  otpu_vpu #(.LANES(L), .WBUF(WBUF), .HAS_SE(1'b1), .ONE_TREE(ONE_TREE), .COMP8(COMP8)) dut (
    .clk, .rst, .start, .cmd, .rdy, .done, .gnt, .ren, .ta_en, .ta_addr, .ta_data, .tb_en,
    .tb_addr, .tb_data, .tw_en, .tw_addr, .tw_data, .pf_u, .pf_frz,
    .ss_req, .ss_gnt, .ss_cfg, .ss_pe, .ss_in_v, .ss_in_d, .ss_fk, .ss_fi, .ss_fd, .ss_y_v,
    .ss_y_d, .ss_o_v, .ss_o_d);

  // TMEM: reads on the lanes' enable, the data the next cycle (held otherwise); writes on the
  // grant. The grant is random (WBUF: it only takes the write buffer's head).
  logic [31:0] tm [TW];
  always_ff @(posedge clk) begin
    for (int l = 0; l < L; l++) begin
      if (ren && ta_en[l]) ta_data[l] <= tm[ta_addr[l][15:0]];
      if (ren && tb_en[l]) tb_data[l] <= tm[tb_addr[l][15:0]];
      if (gnt && tw_en[l]) tm[tw_addr[l][15:0]] <= tw_data[l];
    end
    gnt <= ($urandom % 100) < GNT_PCT;
  end

  // the outputs, as the DMA takes them (qualified by SE)
  int fout, ny = 0, no = 0;
  always_ff @(posedge clk) begin
    if (ss_y_v) begin
      $fdisplay(fout, "Y %08h %08h %08h %08h %08h %08h %08h %08h", ss_y_d[0], ss_y_d[1],
                ss_y_d[2], ss_y_d[3], ss_y_d[4], ss_y_d[5], ss_y_d[6], ss_y_d[7]);
      ny <= ny + 1;
    end
    if (ss_o_v) begin
      $fdisplay(fout, "O %08h", ss_o_d);
      no <= no + 1;
    end
  end

  // the VOPs: started in order while released (vrel), as the sequencer starts them (a
  // register set from rdy: a start can come the cycle after rdy falls); done counted
  cmd_t vq [MAXV];
  int nvop = 0, vrel = 0, vst = 0, vdn = 0;
  always_ff @(posedge clk) begin
    start <= 1'b0;
    if (!rst && !start && vst < vrel && vst < nvop && rdy) begin
      start <= 1'b1;
      cmd <= vq[vst];
      vst <= vst + 1;
    end
  end
  always_ff @(posedge clk) if (done) vdn <= vdn + 1;

  // the rules: no TMEM access and no VOP unfinished in stream mode; no output outside it
  always_ff @(posedge clk) if (!rst) begin
    if (ss_gnt && (ta_en != '0 || tb_en != '0 || tw_en != '0))
      $fatal(1, "TMEM access in stream mode at %0t", $time);
    if (ss_gnt && vdn != vst)
      $fatal(1, "granted with %0d VOPs unfinished at %0t", vst - vdn, $time);
    if (!ss_gnt && (ss_y_v || ss_o_v)) $fatal(1, "stream output outside the grant at %0t", $time);
    if (ss_gnt && (done || start)) $fatal(1, "VOP start/done in stream mode at %0t", $time);
  end

  logic [31:0] seg [MAXSEG][L];
  int fin, n, hdr, i, nfill, nseg, pct, nxt, rows_q, qen_q, fk_i, fi_i, spin, y0, o0, dly;
  int h [13];
  logic [31:0] w [L], op, fl;
  string fname_in, fname_out, fname_tm, fname_dump;
  initial begin
    if (!$value$plusargs("in=%s", fname_in)) $fatal(1, "need +in=");
    if (!$value$plusargs("out=%s", fname_out)) $fatal(1, "need +out=");
    if (!$value$plusargs("dump=%s", fname_dump)) $fatal(1, "need +dump=");
    for (int a = 0; a < TW; a++) tm[a] = '0;
    if ($value$plusargs("tmem=%s", fname_tm)) $readmemh(fname_tm, tm);
    fin = $fopen(fname_in, "r");
    fout = $fopen(fname_out, "w");
    if (fin == 0 || fout == 0) $fatal(1, "cannot open the files");
    for (int l = 0; l < L; l++) begin ss_fd[l] = '0; ss_in_d[l] = '0; end
    ss_cfg = '0;
    n = $fscanf(fin, "%h", nvop);
    if (nvop > MAXV) $fatal(1, "too many VOPs");
    for (i = 0; i < nvop; i++) begin
      n = $fscanf(fin, "%h %h", op, fl);
      vq[i] = '0;
      vq[i].op = op[7:0];
      vq[i].flags = fl[7:0];
      n = $fscanf(fin, "%h %h %h %h %h %h %h", vq[i].w1, vq[i].w2, vq[i].w3, vq[i].w4, vq[i].w5,
                  vq[i].w6, vq[i].w7);
    end
    repeat (4) @(negedge clk);
    rst = 1'b0;
    forever begin
      n = $fscanf(fin, "%h", hdr);
      if (n != 1 || hdr == 0) break;
      for (i = 0; i < 13; i++) n = $fscanf(fin, "%h", h[i]);
      nfill = h[7]; nseg = h[8]; pct = h[9]; dly = h[11];
      if (nseg > MAXSEG) $fatal(1, "too many segments");
      // release VOPs, then ask for SE a little later (they may be queued or in flight)
      vrel = h[10];
      repeat (dly) @(negedge clk);
      ss_cfg.ns = 6'(h[0]); ss_cfg.rows = 9'(h[1]); ss_cfg.a_en = h[2][0];
      ss_cfg.a_sel = h[3][0]; ss_cfg.dmode = 2'(h[4]); ss_cfg.g_src = 2'(h[5]);
      ss_cfg.q_en = h[6][0]; ss_cfg.pad64 = h[12][0];
      rows_q = h[1]; qen_q = h[6];
      ss_req = 1'b1;
      spin = 0;
      while (!ss_gnt) begin
        @(negedge clk);
        if (++spin > 100000) $fatal(1, "no grant");
      end
      for (i = 0; i < nfill; i++) begin
        n = $fscanf(fin, "%h %h", fk_i, fi_i);
        for (int l = 0; l < L; l++) n = $fscanf(fin, "%h", w[l]);
        ss_fk = 3'(fk_i); ss_fi = 5'(fi_i);
        for (int l = 0; l < L; l++) ss_fd[l] = w[l];
        @(negedge clk);
      end
      ss_fk = SF_NONE;
      for (i = 0; i < nseg; i++)
        for (int l = 0; l < L; l++) n = $fscanf(fin, "%h", seg[i][l]);
      y0 = ny; o0 = no;
      nxt = 0;
      spin = 0;
      // the stream: pe at random, segments on consecutive pe cycles, then bubbles until every
      // output is out
      while (ny - y0 < nseg || (qen_q != 0 && no - o0 < rows_q)) begin
        ss_pe = ($urandom % 100) < pct;
        ss_in_v = ss_pe && nxt < nseg;
        for (int l = 0; l < L; l++) ss_in_d[l] = (nxt < nseg) ? seg[nxt][l] : 32'hDEAD_BEEF;
        if (ss_in_v) nxt++;
        @(negedge clk);
        if (++spin > 100000) $fatal(1, "stream hangs: %0d/%0d segments, %0d/%0d o", ny - y0,
                                    nseg, no - o0, rows_q);
      end
      ss_pe = 1'b0;
      ss_in_v = 1'b0;
      // a few more pe cycles: nothing else may come out
      for (i = 0; i < 300; i++) begin
        ss_pe = ($urandom % 2) == 0;
        @(negedge clk);
      end
      ss_pe = 1'b0;
      @(negedge clk);
      if (ny - y0 != nseg || no - o0 != (qen_q != 0 ? rows_q : 0))
        $fatal(1, "extra outputs: %0d/%0d segments, %0d/%0d o", ny - y0, nseg, no - o0, rows_q);
      $fdisplay(fout, "E");
      ss_req = 1'b0;
      @(negedge clk);
    end
    // the rest of the VOPs, then their writes
    vrel = nvop;
    spin = 0;
    while (vdn < nvop) begin
      @(negedge clk);
      if (++spin > 2000000) $fatal(1, "VOPs hang: %0d/%0d done", vdn, nvop);
    end
    repeat (20) @(negedge clk);
    $writememh(fname_dump, tm);
    $fdisplay(fout, "T");
    $fclose(fout);
    $finish;
  end
endmodule
