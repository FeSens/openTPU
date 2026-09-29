// Unit test of rtl/vpu/otpu_se_comp.sv (tests/test_se_comp.py): chunks of composite functions
// issued the way the VPU issues them (a decision HA cycles before S0, held by `hold`; the
// latency rule: a chunk never enters while chunks with more passes are in flight), with
// random gaps and random `en` stalls. EXT = 1 puts the stage units outside, built the way SE's
// owners build them: stage 0 the VPU's slot 0 (otpu_fmma, e = 1), stage 1 the tail's U (the
// same), stage 2 the tail's Q (fmul, and the adder's other operand through the partial loop's
// feedback register), further stages as stage 1.
// Input (+in=, hex): "n issue_pct en_pct seed", then n lines "func x0..x(L-1) y0..y(L-1)".
// Output (+out=): per chunk, in the order they finish, "tag d0 .. d(L-1)"; then "E cycles n",
// n the lanes that differ from the RTL's own definitions of the functions (otpu_fp: fp_exp2,
// fp_recip, fp_rsqrt, fp_log2), NaN inputs included.
module tb_se_comp;
  import otpu_pkg::*;
  import otpu_fp::*;
  parameter int LANES = 8;
  parameter int NS    = 3;
  parameter int EXT   = 0;
  parameter int HA    = 2;
  localparam int L = LANES, MW = 16, LM = 2, LA = 4;

  logic clk = 1'b0, rst = 1'b1, en = 1'b0;
  always #5 clk = !clk;

  logic          in_v, hold, out_v;
  logic [2:0]    in_c;
  logic [MW-1:0] in_meta, out_meta;
  logic [L-1:0]  in_m, out_m;
  f32_t          in_a [L], in_b [L], out_d [L];
  logic [NS-1:0] st_sel;
  f32_t          st_a [NS][L], st_b [NS][L], st_c [NS][L], st_e [NS][L], st_y [NS][L];

  otpu_se_comp #(.LANES(L), .NS(NS), .MW(MW), .HA(HA), .EXT(EXT)) dut (
    .clk, .rst, .en, .in_v, .in_c, .in_a, .in_b, .in_m, .in_meta, .hold, .st_sel, .st_a, .st_b,
    .st_c, .st_e, .st_y, .out_v, .out_d, .out_m, .out_meta);

  if (EXT) begin : g_units
    for (genvar s = 0; s < NS; s++) begin : g_s
      for (genvar l = 0; l < L; l++) begin : g_l
        if (s == 2) begin : g_q
          // Q: fmul on registered operands; the adder's other operand is the partial loop's
          // feedback register, loaded with c LM cycles after the operands
          f32_t ra, rb, rc, cd, fbq, tq;
          always_ff @(posedge clk) if (en) begin
            ra <= st_a[s][l]; rb <= st_b[s][l]; rc <= st_c[s][l];
            fbq <= cd;
          end
          otpu_delay #(.W(32), .N(LM - 1)) u_cd (.clk, .en, .d(rc), .q(cd));
          otpu_fmul #(.LAT(LM)) u_m (.clk, .en, .a(ra), .b(rb), .y(tq));
          otpu_fadd #(.LAT(LA)) u_ad (.clk, .en, .a(fbq), .b(tq), .y(st_y[s][l]));
        end else begin : g_mma
          // slot 0 / U: a*b + c*e with e = 1.0
          f32_t ra, rb, rc, re;
          always_ff @(posedge clk) if (en) begin
            ra <= st_a[s][l]; rb <= st_b[s][l]; rc <= st_c[s][l]; re <= st_e[s][l];
          end
          otpu_fmma #(.LM(LM), .LA(LA)) u_ma (.clk, .en, .a(ra), .b(rb), .c(rc), .e(re),
                                              .y(st_y[s][l]));
        end
      end
    end
  end else begin : g_nounits
    for (genvar s = 0; s < NS; s++) begin : g_s
      for (genvar l = 0; l < L; l++) begin : g_l
        assign st_y[s][l] = '0;
      end
    end
  end

  function automatic int n_pass(input logic [7:0] f);
    int n;
    case (f)
      V_EXP2, V_EXP2SUB: n = 9;
      V_RECIP:           n = 6;
      default:           n = 10;
    endcase
    return (n + NS - 1) / NS;
  endfunction

  // chunks
  int          n, ipct, epct, seed;
  logic [7:0]  fn [];
  logic [31:0] xs [], ys [];
  // issue: a decision HA cycles before S0 (pv[0] the cycle after it, pv[HA-1] at S0)
  logic        pv [HA];
  int          pi [HA];
  int          nxt, nout, inflight, plast, nref;
  longint      cyc;
  assign in_v = pv[HA - 1];
  assign in_c = f_cc(fn[pi[HA - 1]]);
  assign in_meta = MW'(pi[HA - 1]);
  assign in_m = L'(pi[HA - 1] * 37);             // the lane mask is carried, not used
  for (genvar l = 0; l < L; l++) begin : g_in
    assign in_a[l] = xs[pi[HA - 1] * L + l];
    assign in_b[l] = ys[pi[HA - 1] * L + l];
  end

  int fout;
  always @(posedge clk) begin
    if (rst) begin
      for (int j = 0; j < HA; j++) begin pv[j] <= 1'b0; pi[j] <= 0; end
      nxt <= 0; nout <= 0; inflight <= 0; plast <= 0; cyc <= 0; nref <= 0;
    end else begin
      cyc <= cyc + 1;
      if (en) begin
        int infl;
        infl = inflight;
        for (int j = HA - 1; j > 0; j--) begin pv[j] <= pv[j - 1]; pi[j] <= pi[j - 1]; end
        pv[0] <= 1'b0;
        if (out_v) begin
          string sline;
          if (int'(out_meta) != (nout & 16'hFFFF) || out_m != L'(nout * 37))
            $fatal(1, "out of order: %0d, expected %0d", out_meta, nout);
          sline = $sformatf("%0d", nout);
          for (int l = 0; l < L; l++) begin
            f32_t x, y, e;
            sline = {sline, $sformatf(" %08h", out_d[l])};
            // the RTL's own definitions (otpu_fp), NaN inputs included
            x = xs[nout * L + l];
            y = ys[nout * L + l];
            case (fn[nout])
              V_EXP2:    e = fp_exp2(x);
              V_EXP2SUB: e = fp_exp2(fp_sub(x, y));
              V_RECIP:   e = fp_recip(x);
              V_RSQRT:   e = fp_rsqrt(x);
              default:   e = fp_log2(x);
            endcase
            // (fp_exp2 does not canonicalize a NaN before its range checks: -NaN gives +0
            // there, +inf in the VPU's chains and in fp32.py, whose NaN is always positive)
            if ((fn[nout] == V_EXP2 || fn[nout] == V_EXP2SUB) && (is_nan(x) || is_nan(y)))
              e = out_d[l];
            if (e !== out_d[l]) begin
              if (nref < 10) $display("RTLREF chunk %0d lane %0d f=%0d x=%08h y=%08h exp=%08h got=%08h",
                                      nout, l, fn[nout], x, y, e, out_d[l]);
              nref <= nref + 1;
            end
          end
          $fdisplay(fout, "%s", sline);
          nout <= nout + 1;
          infl = infl - 1;
        end
        if (!hold && nxt < n && ($urandom % 100) < ipct &&
            (infl == 0 || n_pass(fn[nxt]) >= plast)) begin
          pv[0] <= 1'b1;
          pi[0] <= nxt;
          plast <= n_pass(fn[nxt]);
          nxt <= nxt + 1;
          infl = infl + 1;
        end
        inflight <= infl;
      end
    end
  end
  always @(negedge clk) en <= !rst && (($urandom % 100) >= epct);

  string fname_in, fname_out;
  int fin, r;
  initial begin
    if (!$value$plusargs("in=%s", fname_in)) $fatal(1, "need +in=");
    if (!$value$plusargs("out=%s", fname_out)) $fatal(1, "need +out=");
    fin = $fopen(fname_in, "r");
    fout = $fopen(fname_out, "w");
    if (fin == 0 || fout == 0) $fatal(1, "cannot open the files");
    r = $fscanf(fin, "%h %h %h %h", n, ipct, epct, seed);
    if (r != 4) $fatal(1, "bad header");
    void'($urandom(seed));
    fn = new[n + 1];
    xs = new[(n + 1) * L];
    ys = new[(n + 1) * L];
    for (int i = 0; i < n; i++) begin
      r = $fscanf(fin, "%h", fn[i]);
      for (int l = 0; l < L; l++) r = $fscanf(fin, "%h", xs[i * L + l]);
      for (int l = 0; l < L; l++) r = $fscanf(fin, "%h", ys[i * L + l]);
      if (r != 1) $fatal(1, "bad chunk %0d", i);
    end
    fn[n] = V_EXP2;
    for (int l = 0; l < L; l++) begin xs[n * L + l] = '0; ys[n * L + l] = '0; end
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst = 1'b0;
    while (nout < n) begin
      @(posedge clk);
      if (cyc > 64'd200 * n + 10000) $fatal(1, "timeout: %0d of %0d chunks out", nout, n);
    end
    @(posedge clk);
    $fdisplay(fout, "E %0d %0d", cyc, nref);
    $fclose(fout);
    $finish;
  end
endmodule
