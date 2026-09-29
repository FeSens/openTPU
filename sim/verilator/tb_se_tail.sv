// Unit test of rtl/vpu/otpu_se_tail.sv (tests/test_se_tail.py): the tail with a stand-in for
// the VPU's slot-0 partial loop (RDOT: pacc = S * xa + prev through a registered multiply-add,
// TA = 1 + LM + LA from X) and its folding tree (otpu_vtree), fed streams with random pe stalls,
// back to back (pe stops after a stream's last output, as the DMA stops it, or after a drain).
// Input (+in=, hex words): per stream a header
//   1 ns rows a_en a_sel dmode g_src q_en nfill nseg pe_pct
// then nfill lines "fk fi d0 .. d7" and nseg lines "d0 .. d7" (the stream's segments); a 0
// ends the file. Output (+out=): per stream a line "Y d0 .. d7" per updated segment and "O d"
// per row output, in order, then "E".
module tb_se_tail;
  import otpu_pkg::*;
  import otpu_fp::*;
  localparam int L = 8, LM = 2, LA = 4, TA = 1 + LM + LA;
  localparam int MAXSEG = 256 * 32;

  logic clk = 1'b0, rst = 1'b1, init = 1'b0;
  always #5 clk = !clk;
  ss_cfg_t cfg;
  logic [2:0] fk = SF_NONE;
  logic [4:0] fi = '0;
  f32_t fd [L];
  logic pe = 1'b0, in_v = 1'b0;
  f32_t in_d [L];
  f32_t xd [L], xa [L];
  ss_meta_t xm;
  f32_t kv, o_d;
  logic kv_v, y_v, o_v;
  f32_t y_d [L];

  otpu_se_tail #(.LANES(L), .TA(TA)) dut (.clk, .rst, .init, .cfg, .fk, .fi, .fd, .pe, .in_v,
                                          .in_d, .xd, .xa, .xm, .kv, .kv_v, .y_v, .y_d, .o_v,
                                          .o_d);

  // VPU slot 0 in RDOT mode: ra/rb/rc/re -> a*b + c*e, c the partial RL segments ago (+0 at a
  // row's first segments, cleared as it enters rc), e = 1
  f32_t pacc [L];
  for (genvar l = 0; l < L; l++) begin : g_s0
    f32_t ra, rb, rc, fbq;
    always_ff @(posedge clk) if (pe) begin
      ra <= xd[l];
      rb <= xa[l];
      rc <= (xm.v && xm.first) ? F_ZERO : fbq;
      fbq <= pacc[l];
    end
    otpu_fmma #(.LM(LM), .LA(LA)) u_ma (.clk, .en(pe), .a(ra), .b(rb), .c(rc), .e(F_ONE),
                                        .y(pacc[l]));
  end
  ss_meta_t mt;
  otpu_delay #(.W($bits(ss_meta_t)), .N(TA)) u_mt (.clk, .en(pe), .d(xm), .q(mt));
  otpu_vtree #(.LANES(L), .LA(LA)) u_vt (.clk, .rst(rst || init), .en(pe), .pacc,
                                         .cap(mt.v && mt.final_), .row_last(mt.row_last),
                                         .sub(mt.sub), .root(kv), .root_v(kv_v));

  // the outputs, as the DMA takes them
  int fout, ny = 0, no = 0;
  always_ff @(posedge clk) begin
    if (pe && y_v) begin
      $fdisplay(fout, "Y %08h %08h %08h %08h %08h %08h %08h %08h", y_d[0], y_d[1], y_d[2],
                y_d[3], y_d[4], y_d[5], y_d[6], y_d[7]);
      ny <= ny + 1;
    end
    if (pe && o_v) begin
      $fdisplay(fout, "O %08h", o_d);
      no <= no + 1;
    end
  end

  logic [31:0] seg [MAXSEG][L];
  int fin, n, hdr, i, nfill, nseg, pct, nxt, rows_q, qen_q, fk_i, fi_i, spin, y0, o0;
  int h [10];
  logic [31:0] w [L];
  string fname_in, fname_out;
  initial begin
    if (!$value$plusargs("in=%s", fname_in)) $fatal(1, "need +in=");
    if (!$value$plusargs("out=%s", fname_out)) $fatal(1, "need +out=");
    fin = $fopen(fname_in, "r");
    fout = $fopen(fname_out, "w");
    if (fin == 0 || fout == 0) $fatal(1, "cannot open the files");
    for (int l = 0; l < L; l++) begin fd[l] = '0; in_d[l] = '0; end
    cfg = '0;
    repeat (4) @(negedge clk);
    rst = 1'b0;
    forever begin
      n = $fscanf(fin, "%h", hdr);
      if (n != 1 || hdr == 0) break;
      for (i = 0; i < 10; i++) n = $fscanf(fin, "%h", h[i]);
      nfill = h[7]; nseg = h[8]; pct = h[9];
      if (nseg > MAXSEG) $fatal(1, "too many segments");
      cfg.ns = 6'(h[0]); cfg.rows = 9'(h[1]); cfg.a_en = h[2][0]; cfg.a_sel = h[3][0];
      cfg.dmode = 2'(h[4]); cfg.g_src = 2'(h[5]); cfg.q_en = h[6][0];
      rows_q = h[1]; qen_q = h[6];
      @(negedge clk);
      @(negedge clk);
      init = 1'b1;
      @(negedge clk);
      init = 1'b0;
      for (i = 0; i < nfill; i++) begin
        n = $fscanf(fin, "%h %h", fk_i, fi_i);
        for (int l = 0; l < L; l++) n = $fscanf(fin, "%h", w[l]);
        fk = 3'(fk_i); fi = 5'(fi_i);
        for (int l = 0; l < L; l++) fd[l] = w[l];
        @(negedge clk);
      end
      fk = SF_NONE;
      for (i = 0; i < nseg; i++)
        for (int l = 0; l < L; l++) n = $fscanf(fin, "%h", seg[i][l]);
      y0 = ny; o0 = no;                 // the outputs so far (counted by the monitor)
      nxt = 0;
      spin = 0;
      @(negedge clk);
      // the stream: pe at random (pct percent of the cycles), segments on consecutive pe
      // cycles, then bubbles until every output is out
      while (ny - y0 < nseg || (qen_q != 0 && no - o0 < rows_q)) begin
        pe = ($urandom % 100) < pct;
        in_v = pe && nxt < nseg;
        for (int l = 0; l < L; l++) in_d[l] = (nxt < nseg) ? seg[nxt][l] : 32'hDEAD_BEEF;
        if (in_v) nxt++;
        @(negedge clk);
        if (++spin > 100000) $fatal(1, "stream hangs: %0d/%0d segments, %0d/%0d o", ny - y0,
                                    nseg, no - o0, rows_q);
      end
      in_v = 1'b0;
      // half the streams: a few more pe cycles, where nothing else may come out; the others
      // stop at once, as the DMA does (one more pe at most), leaving what the pipeline still
      // holds to the next stream (which must not see it)
      if (($urandom % 2) == 0) begin
        for (i = 0; i < 300; i++) begin
          pe = ($urandom % 2) == 0;
          @(negedge clk);
        end
      end else begin
        pe = ($urandom % 2) == 0;
        @(negedge clk);
      end
      pe = 1'b0;
      if (ny - y0 != nseg || no - o0 != (qen_q != 0 ? rows_q : 0))
        $fatal(1, "extra outputs: %0d/%0d segments, %0d/%0d o", ny - y0, nseg, no - o0, rows_q);
      $fdisplay(fout, "E");
    end
    $fclose(fout);
    $finish;
  end
endmodule
