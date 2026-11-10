// EXPERIMENTAL (otpu_ppk.sv): ppk_core against an integer model of the WG 8 column's arithmetic.
// Random chunks (two int8 rows, NCOL lanes of weight bytes at the format WF, the four scale
// vectors), rows of 1-5 blocks; every column's four sums checked in the cycle the schedule says
// (chunk t driven in cycle t: row r of column j in cycle t + 17 + HOLD + r + j). Two clocks from
// one process, every clk_s rising edge on a clk_f rising edge (as the MMCM's). Prints
// TB_RESULT PASS / FAIL. Verilator (run_vl.sh) with the behavioural DSP48E2, or xsim with AMD's
// (run_xsim.sh).
module tb_ppk #(
  parameter int NCOL = 1,
  parameter bit BF = 1'b1,
  parameter bit HOLD = 1'b0,
  parameter int WF = 3,               // 0 int8, 1 int4, 2 / 3 the 4-bit floats (otpu_mxu's wf)
  parameter int NCHUNK = 300,
  parameter int SEED = 1
);
  localparam int D = 128;
  localparam int T0 = 40;             // the first chunk's cycle: the reset done, and past the
                                      // unisims' GSR (100 ns; a slow cycle is 4 time units)
  localparam int LAT = 17 + (HOLD ? 1 : 0);
  logic clk_s = 1'b0, clk_f = 1'b0, rst = 1'b1;
  logic [D-1:0][7:0]           x0_in, x1_in;
  logic [NCOL-1:0][D-1:0][7:0] w_in;
  logic [1:0]                  wf_in;
  logic [3:0][7:0][16:0]       bm_in;
  logic                        first_in;
  logic [NCOL-1:0][3:0][47:0]  acc_out;

  ppk_core #(.NCOL(NCOL), .D(D), .BF(BF), .HOLD(HOLD)) dut (
    .clk_s, .clk_f, .rst, .x0_in, .x1_in, .w_in, .wf_in, .bm_in, .first_in, .acc_out);

  initial begin
    int n;
    n = 0;
    forever begin
      #1;
      n++;
      clk_f = (n % 2) == 1;
      clk_s = (((n + 1) / 2) % 2) == 1;
    end
  end

  // xorshift32: the same stream on every simulator
  int unsigned rs = 32'h9e3779b9 ^ SEED;
  function automatic int unsigned rnd();
    rs ^= rs << 13;
    rs ^= rs >> 17;
    rs ^= rs << 5;
    return rs;
  endfunction
  // the decoded weights (otpu_mxu's wdec, written out)
  function automatic int sx4(int c);
    return ((c & 8) != 0) ? c - 16 : c;
  endfunction
  function automatic int d4(int c, bit fp);
    int mag;
    if (!fp) return sx4(c);
    case (c & 7)
      0: mag = 0;  1: mag = 1;  2: mag = 2;  3: mag = 3;
      4: mag = 4;  5: mag = 6;  6: mag = 8;  default: mag = 12;
    endcase
    return ((c & 8) != 0) ? -mag : mag;
  endfunction
  function automatic int dlo(int b);
    return (WF == 0) ? (b & 15) : d4(b & 15, WF != 1);
  endfunction
  function automatic int dhi(int b);
    return (WF == 0) ? sx4((b >> 4) & 15) : d4((b >> 4) & 15, WF != 1);
  endfunction

  byte              xm [2][NCHUNK][D];
  byte unsigned     wm [NCHUNK][NCOL][D];
  int               bmm [NCHUNK][4][8];
  bit               fm [NCHUNK];
  longint           ex [NCHUNK][NCOL][4];
  localparam longint M48 = (64'd1 << 48) - 1;

  initial begin
    int rem;
    rem = 0;
    for (int t = 0; t < NCHUNK; t++) begin
      for (int r = 0; r < 2; r++)
        for (int i = 0; i < D; i++) xm[r][t][i] = byte'(rnd());
      for (int j = 0; j < NCOL; j++)
        for (int i = 0; i < D; i++) wm[t][j][i] = byte'(rnd());
      for (int q = 0; q < 4; q++)
        for (int g = 0; g < 8; g++) bmm[t][q][g] = int'(rnd() % (1 << 17));
      fm[t] = (rem == 0);
      rem = (rem == 0) ? int'(rnd() % 5) : rem - 1;
    end
    // the model: per row r and group g, S0 = sum x * lo, S1 = sum x * hi; u_ce takes S0 (int8:
    // S0 + 16 S1) with bm[2r], u_ce2 S1 with bm[2r + 1]; the sums accumulate over a row's blocks
    for (int t = 0; t < NCHUNK; t++)
      for (int j = 0; j < NCOL; j++)
        for (int r = 0; r < 2; r++) begin
          longint lo, hi;
          lo = 0;
          hi = 0;
          for (int g = 0; g < 8; g++) begin
            longint s0, s1;
            s0 = 0;
            s1 = 0;
            for (int k = 0; k < 16; k++) begin
              s0 += longint'(xm[r][t][g * 16 + k]) * dlo(int'(wm[t][j][g * 16 + k]));
              s1 += longint'(xm[r][t][g * 16 + k]) * dhi(int'(wm[t][j][g * 16 + k]));
            end
            lo += (s0 + ((WF == 0) ? 16 * s1 : 0)) * bmm[t][2 * r][g];
            hi += s1 * bmm[t][2 * r + 1][g];
          end
          ex[t][j][2 * r] = ((fm[t] || t == 0) ? 64'd0 : ex[t - 1][j][2 * r]) + lo & M48;
          ex[t][j][2 * r + 1] = ((fm[t] || t == 0) ? 64'd0 : ex[t - 1][j][2 * r + 1]) + hi & M48;
        end
  end

  initial begin
    int checks, bad;
    checks = 0;
    bad = 0;
    wf_in = 2'(WF);
    x0_in = '0; x1_in = '0; w_in = '0; bm_in = '0; first_in = 1'b1;
    @(posedge clk_s);
    for (int c = 0; c < T0 + NCHUNK + LAT + NCOL + 4; c++) begin
      @(negedge clk_s);
      if (c == 4) rst = 1'b0;
      // the sums registered in this cycle
      for (int j = 0; j < NCOL; j++)
        for (int r = 0; r < 2; r++) begin
          int t;
          t = c - T0 - LAT - r - j;
          if (t >= 0 && t < NCHUNK)
            for (int h = 0; h < 2; h++) begin
              longint got;
              got = longint'(acc_out[j][2 * r + h]);
              checks++;
              if (got != ex[t][j][2 * r + h]) begin
                bad++;
                if (bad <= 10)
                  $display("MISMATCH chunk %0d column %0d row %0d half %0d: %h, expected %h",
                           t, j, r, h, got, ex[t][j][2 * r + h]);
              end
            end
        end
      // the chunk for the next edge
      if (c >= T0 && c - T0 < NCHUNK) begin
        int t;
        t = c - T0;
        for (int i = 0; i < D; i++) begin
          x0_in[i] = xm[0][t][i];
          x1_in[i] = xm[1][t][i];
          for (int j = 0; j < NCOL; j++) w_in[j][i] = wm[t][j][i];
        end
        for (int q = 0; q < 4; q++)
          for (int g = 0; g < 8; g++) bm_in[q][g] = 17'(bmm[t][q][g]);
        first_in = fm[t] || t == 0;
      end else begin
        x0_in = '0; x1_in = '0; w_in = '0; bm_in = '0; first_in = 1'b1;
      end
    end
    if (bad == 0 && checks == NCHUNK * NCOL * 4)
      $display("TB_RESULT PASS NCOL %0d BF %0d HOLD %0d WF %0d: %0d sums", NCOL, BF, HOLD, WF,
               checks);
    else
      $display("TB_RESULT FAIL NCOL %0d BF %0d HOLD %0d WF %0d: %0d of %0d sums wrong", NCOL, BF,
               HOLD, WF, bad, checks);
    $finish;
  end
endmodule
