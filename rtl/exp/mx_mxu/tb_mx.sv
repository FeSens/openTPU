// EXPERIMENTAL (docs/mx_mxu.md): xsim testbench of mx_top on the unisim DSP models (7-series
// pumped DSP48E1s or UltraScale+ packed DSP48E2s: the family file compiled with it). Random rows
// of 1..8 blocks: int8 / int4 / E2M1 weights (both streams), random scale words (dw, 6-bit m_b),
// the rows' exponent words (da) with flushed terms (sh < 0) and full shifts (sh = Wf), W4X and
// W8X, PAIR (the low rows in the first half of the columns). Checked:
//   - every row's 48-bit accumulators at the result FIFO's input against a longint reference
//     (sh = Wf - dw - da, B = m << sh or 0, acc = sum B * c; PAIR: row j + row j + ROWS/2);
//   - every lane output, y = old * alpha + (i2f48(acc) * Sw) * Sa', against the reference with
//     an independent i2f48 (round to nearest even) and the otpu_fp multiply / add;
//   - a separate lane fed with edge-case integers (powers of two, ties, all ones, +-2^47 - 1):
//     y = i2f48(x) exactly.
// Run: rtl/exp/mx_mxu/run_xsim.sh (the build host).
`timescale 1ps / 1ps
module tb_mx;
  import otpu_fp::*;
  localparam int NCOL = 2, ROWS = 2 * NCOL, E = 2, D = 128;
  logic clk_in = 1'b0;
  always #3750 clk_in = ~clk_in;        // 133.33 MHz
  logic rst = 1'b1;
  logic [ROWS-1:0][D-1:0][7:0] x_in;
  logic [D-1:0][7:0] chunk_in;
  logic [1:0] wf_in;
  logic h_in;
  logic [31:0] sw_lo_in, sw_hi_in, swr_in;
  logic [ROWS-1:0][31:0] da_in, sa_in;
  logic [NCOL-1:0] hsel_in;
  logic pair_in, fmt_in, v_in, first_in, last_in;
  logic [7:0] m_in;
  logic [E-1:0][31:0] old_in, alpha_in, y;
  logic [E-1:0] yv;
  logic [E-1:0][23:0] ytag;
  logic busy;
  mx_top #(.NCOL(NCOL), .E(E), .D(D)) dut (.*);
  wire clk = dut.clk_s;

  // ---------------------------------------------------------------- reference
  function automatic logic [7:0] dec4(input logic [3:0] c, input logic fp);
    if (!fp) return {{4{c[3]}}, c};
    case (c)
      4'd5: return 8'd6;   4'd6: return 8'd8;   4'd7: return 8'd12;  4'd8: return 8'd0;
      4'd9: return -8'sd1; 4'd10: return -8'sd2; 4'd11: return -8'sd3; 4'd12: return -8'sd4;
      4'd13: return -8'sd6; 4'd14: return -8'sd8; 4'd15: return -8'sd12;
      default: return 8'(c);
    endcase
  endfunction
  function automatic logic signed [7:0] wref(input logic [D-1:0][7:0] ch, input int i,
                                             input logic hb, input logic [1:0] wf);
    logic [D*8-1:0] w;
    w = ch;
    if (wf == 2'd0) return w[i*8 +: 8];
    return dec4(w[(hb ? D*4 : 0) + 4*i +: 4], wf == 2'd2);
  endfunction
  // i2f of a 48-bit integer, round to nearest even (written apart from the RTL's)
  function automatic f32_t i2f48_ref(input logic signed [47:0] x);
    longint unsigned a, mant, rem, half;
    int p, sh;
    logic s;
    if (x == 0) return 32'h0;
    s = x < 0;
    a = s ? longint'(-x) : longint'(x);
    p = 0;
    for (int b = 0; b < 48; b++) if (a[b]) p = b;
    if (p <= 23) mant = a << (23 - p);
    else begin
      sh = p - 23;
      mant = a >> sh;
      rem = a & ((64'd1 << sh) - 1);
      half = 64'd1 << (sh - 1);
      if (rem > half || (rem == half && mant[0])) mant = mant + 1;
      if (mant == (64'd1 << 24)) begin
        mant = mant >> 1;
        p = p + 1;
      end
    end
    return {s, 8'(127 + p), mant[22:0]};
  endfunction

  typedef logic signed [47:0] a48_t;
  a48_t   racc [$][ROWS];          // expected rows at the FIFO input
  f32_t   ry [E][$];               // expected lane outputs
  logic [23:0] rtag [E][$];
  int rows_sent = 0;

  // per-command state
  logic cpair, cfmt;
  logic [NCOL-1:0] chsel;
  int cm;
  f32_t csa [ROWS];
  f32_t cold [E], calpha [E];
  int dwmax, damax;

  function automatic f32_t sa_prime(input f32_t s, input logic fmt);
    logic [8:0] e;
    e = 9'(s[30:23]) - (fmt ? 9'd16 : 9'd11);
    return (e[8] || e == 0) ? {s[31], 31'd0} : {s[31], e[7:0], s[22:0]};
  endfunction
  function automatic f32_t rnd_f32();
    // a positive or negative normal value with a moderate exponent
    return {1'($urandom), 8'(100 + $urandom_range(0, 50)), 23'($urandom)};
  endfunction

  task automatic idle(input int n);
    for (int c = 0; c < n; c++) begin
      @(posedge clk); #100;
      v_in = 1'b0; first_in = 1'b0; last_in = 1'b0;
    end
  endtask

  // one weight row of kb blocks
  task automatic send_row(input int kb, input logic [1:0] wf);
    a48_t acc [ROWS];
    f32_t swv;
    for (int r = 0; r < ROWS; r++) acc[r] = 0;
    swv = rnd_f32();
    for (int k = 0; k < kb; k++) begin
      logic [D-1:0][7:0] ch;
      logic [ROWS-1:0][D-1:0][7:0] xx;
      logic [31:0] slo, shi;
      logic [ROWS-1:0][31:0] da;
      logic hh;
      for (int i = 0; i < D; i++) ch[i] = 8'($urandom);
      for (int r = 0; r < ROWS; r++)
        for (int i = 0; i < D; i++) begin
          logic [7:0] b;
          b = 8'($urandom);
          if (b == 8'h80) b = 8'h81;                    // q8 saturates to [-127, 127]
          xx[r][i] = b;
        end
      slo = {24'($urandom), 8'($urandom_range(0, dwmax))};
      shi = {24'($urandom), 8'($urandom_range(0, dwmax))};
      for (int r = 0; r < ROWS; r++)
        for (int b = 0; b < 4; b++) da[r][8*b +: 8] = 8'($urandom_range(0, damax));
      hh = (wf != 0) && !cpair && 1'($urandom);
      // reference
      for (int r = 0; r < ROWS; r++) begin
        logic hs;
        logic [31:0] sw;
        hs = chsel[r / 2];
        sw = hs ? shi : slo;
        for (int b = 0; b < 4; b++) begin
          longint c, bb;
          int sh;
          c = 0;
          for (int i = 32 * b; i < 32 * b + 32; i++)
            c += longint'($signed(xx[r][i])) * longint'(hs ? wref(ch, i, 1'b1, wf) : wref(ch, i, hh, wf));
          sh = (cfmt ? 16 : 11) - int'(sw[7:0]) - int'(da[r][8*b +: 8]);
          bb = (sh < 0) ? 0 : (longint'(cfmt ? 1 : sw[8 + 6*b +: 6]) << sh);
          acc[r] = acc[r] + a48_t'(c * bb);
        end
      end
      // drive
      @(posedge clk); #100;
      x_in = xx; chunk_in = ch; wf_in = wf; h_in = hh; sw_lo_in = slo; sw_hi_in = shi;
      da_in = da; v_in = 1'b1; first_in = (k == 0); last_in = (k == kb - 1); swr_in = swv;
    end
    begin
      a48_t res [ROWS];
      for (int j = 0; j < ROWS; j++)
        res[j] = (cpair && j < ROWS / 2) ? acc[j] + acc[j + ROWS / 2] : acc[j];
      racc.push_back(res);
      for (int j = 0; j < cm; j++) begin
        f32_t v1, v2, v3;
        v1 = fp_mul(i2f48_ref(res[j]), swv);
        v2 = fp_mul(v1, sa_prime(csa[j], cfmt));
        v3 = fp_add(fp_mul(cold[j % E], calpha[j % E]), v2);
        ry[j % E].push_back(v3);
        rtag[j % E].push_back({16'(rows_sent), 8'(j)});
      end
      rows_sent++;
    end
    // the drain takes ceil(m / E) cycles a row
    if ((cm + E - 1) / E + 1 > kb) idle((cm + E - 1) / E + 1 - kb);
  endtask

  task automatic command(input logic pair, input logic fmt, input int dwm, input int dam);
    idle(80);                                     // the previous command drained
    cpair = pair; cfmt = fmt; dwmax = dwm; damax = dam;
    for (int p = 0; p < NCOL; p++) chsel[p] = pair ? (p >= NCOL / 2) : 1'($urandom);
    cm = pair ? ROWS / 2 : ROWS;
    for (int r = 0; r < ROWS; r++) csa[r] = rnd_f32();
    for (int k = 0; k < E; k++) begin
      cold[k] = 1'($urandom) ? rnd_f32() : 32'h0;
      calpha[k] = 1'($urandom) ? rnd_f32() : 32'h3F80_0000;
    end
    @(posedge clk); #100;
    hsel_in = chsel; pair_in = pair; fmt_in = fmt; m_in = 8'(cm);
    for (int r = 0; r < ROWS; r++) sa_in[r] = csa[r];
    for (int k = 0; k < E; k++) begin old_in[k] = cold[k]; alpha_in[k] = calpha[k]; end
    idle(4);
  endtask

  // ---------------------------------------------------------------- checks
  int errs = 0, nacc = 0, ny = 0;
  always @(posedge clk) if (!rst && dut.rv === 1'b1) begin
    if (racc.size() == 0) begin
      errs++;
      if (errs < 20) $display("ACC MISMATCH: a row with none expected");
    end else begin
      a48_t ex [ROWS];
      ex = racc.pop_front();
      nacc++;
      for (int j = 0; j < ROWS; j++)
        if (dut.rowp[j] !== ex[j]) begin
          errs++;
          if (errs < 20) $display("ACC MISMATCH row %0d col %0d: %0d, expected %0d", nacc - 1, j,
                                  $signed(dut.rowp[j]), ex[j]);
        end
    end
  end
  always @(posedge clk) if (!rst)
    for (int k = 0; k < E; k++)
      if (yv[k] === 1'b1) begin
        if (ry[k].size() == 0) begin
          errs++;
          if (errs < 20) $display("Y MISMATCH lane %0d: an output with none expected", k);
        end else begin
          f32_t ex;
          logic [23:0] et;
          ex = ry[k].pop_front();
          et = rtag[k].pop_front();
          ny++;
          if (y[k] !== ex || ytag[k] !== et) begin
            errs++;
            if (errs < 20) $display("Y MISMATCH lane %0d tag %h: %h, expected %h (tag %h)", k,
                                    ytag[k], y[k], ex, et);
          end
        end
      end

  // a lane alone: i2f48 edge cases (y = 0 * 0 + (i2f48(x) * 1) * 1)
  logic lv;
  logic [47:0] lx;
  f32_t ly;
  logic lyv;
  logic [23:0] lyt;
  otpu_mx_lane u_l1 (.clk, .v(lv), .x(lx), .sw(32'h3F80_0000), .sa(32'h3F80_0000),
                     .old(32'h0), .alpha(32'h0), .tag(24'd0), .y(ly), .yv(lyv), .ytag(lyt));
  f32_t li2f [$];
  int nl = 0;
  always @(posedge clk) if (lyv === 1'b1) begin
    f32_t ex;
    ex = li2f.pop_front();
    nl++;
    if (ly !== ex) begin
      errs++;
      if (errs < 20) $display("I2F MISMATCH: %h, expected %h", ly, ex);
    end
  end
  task automatic lane_case(input logic signed [47:0] x);
    @(posedge clk); #100;
    lv = 1'b1; lx = x;
    li2f.push_back(i2f48_ref(x));
  endtask

  initial begin
    lv = 1'b0; lx = '0;
    x_in = '0; chunk_in = '0; wf_in = 0; h_in = 0; sw_lo_in = 0; sw_hi_in = 0; swr_in = 0;
    da_in = '0; sa_in = '0; hsel_in = '0; pair_in = 0; fmt_in = 0; v_in = 0; first_in = 0;
    last_in = 0; m_in = 8'(ROWS); old_in = '0; alpha_in = '0;
    cpair = 0; cfmt = 0; chsel = '0; cm = ROWS; dwmax = 3; damax = 3;
    for (int r = 0; r < ROWS; r++) csa[r] = 32'h3F80_0000;
    for (int k = 0; k < E; k++) begin cold[k] = 0; calpha[k] = 0; end
    wait (dut.locked === 1'b1);
    repeat (120) @(posedge clk);                  // the skew lines fill with idle chunks
    #100 rst = 1'b0;
    repeat (8) @(posedge clk);
    // the lane's conversion
    begin
      logic signed [47:0] t [$];
      t = '{48'sd1, -48'sd1, 48'sd16777215, 48'sd16777216, 48'sd16777217, 48'sd16777219,
            48'sd33554431, 48'sd33554435, -48'sd33554433, 48'sh7FFF_FFFF_FFFF,
            -48'sh7FFF_FFFF_FFFF, 48'sh4000_0000_0000, -48'sh8000_0000_0000 + 1,
            48'sh0000_0100_0000_0080, 48'sh0000_0100_0000_0180, 48'sh0000_00FF_FFFF_8000};
      foreach (t[i]) lane_case(t[i]);
      for (int i = 0; i < 400; i++) begin
        logic signed [47:0] r;
        r = 48'(({$urandom, $urandom} & 64'h7FFF_FFFF_FFFF) >> $urandom_range(0, 47));
        lane_case(1'($urandom) ? -r : r);
      end
      @(posedge clk); #100 lv = 1'b0;
    end
    // W4X E2M1, both streams per column, small offsets
    command(1'b0, 1'b0, 3, 3);
    for (int n = 0; n < 40; n++) send_row($urandom_range(1, 8), 2'd2);
    // W4X int4, PAIR
    command(1'b1, 1'b0, 4, 4);
    for (int n = 0; n < 40; n++) send_row($urandom_range(1, 8), 2'd1);
    // W8X int8
    command(1'b0, 1'b1, 6, 6);
    for (int n = 0; n < 40; n++) send_row($urandom_range(1, 8), 2'd0);
    // flush-heavy: offsets past the window (sh < 0) and exactly at it
    command(1'b0, 1'b0, 9, 9);
    for (int n = 0; n < 40; n++) send_row($urandom_range(1, 8), 2'd2);
    command(1'b1, 1'b1, 12, 12);
    for (int n = 0; n < 40; n++) send_row($urandom_range(1, 8), 2'd0);
    // single-block rows back to back (q.K^T-like), E2M1
    command(1'b0, 1'b0, 2, 2);
    for (int n = 0; n < 60; n++) send_row(1, 2'd2);
    idle(150);
    if (racc.size() != 0) begin errs++; $display("ACC MISMATCH: %0d rows never came", racc.size()); end
    for (int k = 0; k < E; k++)
      if (ry[k].size() != 0) begin errs++; $display("Y MISMATCH lane %0d: %0d outputs never came", k, ry[k].size()); end
    if (li2f.size() != 0) begin errs++; $display("I2F MISMATCH: %0d never came", li2f.size()); end
    $display("TB_RESULT %s: %0d rows, %0d lane outputs, %0d conversions, %0d mismatches",
             errs == 0 ? "PASS" : "FAIL", nacc, ny, nl, errs);
    $finish;
  end
endmodule
