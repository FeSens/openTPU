// Synthesis-only cost proxy for 4-bit MXU options (b) and (c) of docs/quant.md: the datapath a
// full-rate 4-bit MXU adds per cycle to the half-rate one (rtl/mxu/otpu_mxu.sv) -- a second
// D-block of weights (the chunk's other half) against a second ACT RAM block, i.e. its nibble
// decode, MCOLS x D products, the pair / group / sub-block tree, i2f, x ws, x ascale, and the
// fp add that pairs the two blocks' terms. Not a working unit (no control, no ACT RAM banking):
// it only measures what the doubled datapath costs. Run with tools/synth/sta.sh.
//   PROD = 0: products in DSP48s, two columns per DSP (as otpu_mxu's tree)   -> option (b)
//   PROD = 1: products in LUTs: a * {0,1,2,3,4,6,8,12} as a mux of a, 3a shifted (E2M1 only) -> (c)
module proxy_mxu_x2
  import otpu_fp::*;
#(
  parameter int D = 128,
  parameter int MCOLS = 2,
  parameter int PROD = 0
) (
  input  logic                  clk,
  input  logic                  en,
  input  logic [D*4-1:0]        w,          // the other half of the chunk (D nibbles)
  input  logic                  fp,         // E2M1 (else int4)
  input  logic [MCOLS*D*8-1:0]  a,          // the second ACT RAM block of every column
  input  logic [15:0]           mb,         // its sub-block multipliers
  input  f32_t                  ws,
  input  f32_t [MCOLS-1:0]      as,
  input  f32_t [MCOLS-1:0]      t_even,     // the first block's terms (from the existing MXU)
  output f32_t [MCOLS-1:0]      t_pair
);
  localparam int SW = 16 + $clog2(D);
  localparam int GS = 16, NG3 = D / GS, GPB = NG3 / 4;
  localparam logic [33:0] PK = 34'd32513;

  function automatic logic [7:0] dec4(input logic [3:0] c, input logic f);
    logic [7:0] mag;
    if (!f) return {{4{c[3]}}, c};
    case (c[2:0])
      3'd5: mag = 8'd6;
      3'd6: mag = 8'd8;
      3'd7: mag = 8'd12;
      default: mag = 8'(c[2:0]);
    endcase
    return c[3] ? -mag : mag;
  endfunction

  logic [D*4-1:0] wr;
  logic fpr;
  logic [MCOLS*D*8-1:0] ar;
  always_ff @(posedge clk) if (en) begin wr <= w; fpr <= fp; ar <= a; end

  logic signed [19:0] s3 [MCOLS][NG3];
  if (PROD == 0) begin : g_dsp
    logic [MCOLS/2-1:0][D-1:0][33:0]   pm;
    logic [MCOLS/2-1:0][D/2-1:0][33:0] pq;
    always_ff @(posedge clk) if (en) begin
      for (int p = 0; p < MCOLS / 2; p++) begin
        for (int i = 0; i < D; i++) begin
          logic signed [24:0] pa;
          pa = $signed({ar[(2*p*D + i)*8 +: 8], 16'b0}) + 25'($signed(ar[((2*p+1)*D + i)*8 +: 8]));
          pm[p][i] <= 34'(pa) * 34'($signed(dec4(wr[4*i +: 4], fpr)));
        end
        for (int q = 0; q < D / 2; q++) pq[p][q] <= pm[p][2*q+1] + (pm[p][2*q] + PK);
      end
      for (int j = 0; j < MCOLS; j++)
        for (int g = 0; g < NG3; g++) begin
          logic signed [19:0] t;
          t = '0;
          for (int k = 0; k < GS / 2; k++)
            if (j % 2 == 1) t = t + 20'(pq[j/2][GS/2*g+k][15:0]);
            else t = t + 20'($signed(pq[j/2][GS/2*g+k][32:16]));
          s3[j][g] <= t;
        end
    end
  end else begin : g_lut
    // a * e for e in {0, 1, 2, 3, 4, 6, 8, 12}: 3a is one adder per activation byte, the rest
    // are shifts; the product is a 4:1 mux of {a, 3a} shifted by 0..2, gated and negated
    logic signed [13:0] pr [MCOLS][D];
    logic signed [15:0] pp [MCOLS][D/2];
    always_ff @(posedge clk) if (en) begin
      for (int j = 0; j < MCOLS; j++)
        for (int i = 0; i < D; i++) begin
          logic signed [13:0] x, x3, m;
          logic [3:0] c;
          c = wr[4*i +: 4];
          x = 14'($signed(ar[(j*D + i)*8 +: 8]));
          x3 = x + (x <<< 1);
          case (c[2:0])                           // E2M1 only (int4 would need 5a, 7a)
            3'd0: m = '0;
            3'd1: m = x;
            3'd2: m = x <<< 1;
            3'd3: m = x3;
            3'd4: m = x <<< 2;
            3'd5: m = x3 <<< 1;
            3'd6: m = x <<< 3;
            default: m = x3 <<< 2;
          endcase
          if (c[3]) m = -m;
          pr[j][i] <= m;
        end
      for (int j = 0; j < MCOLS; j++)
        for (int q = 0; q < D / 2; q++) pp[j][q] <= 16'(pr[j][2*q]) + 16'(pr[j][2*q+1]);
      for (int j = 0; j < MCOLS; j++)
        for (int g = 0; g < NG3; g++) begin
          logic signed [19:0] t;
          t = '0;
          for (int k = 0; k < GS / 2; k++) t = t + 20'(pp[j][GS/2*g+k]);
          s3[j][g] <= t;
        end
    end
  end

  // sub-blocks x multipliers, block sum, i2f (exact: MG <= 24), x ws, x ascale, + t_even
  logic [SW-1:0] u [MCOLS][4], v [MCOLS][4];
  logic signed [SW-1:0] s4 [MCOLS];
  logic [15:0] mr1, mr2;
  always_ff @(posedge clk) if (en) begin
    mr1 <= mb; mr2 <= mr1;
    for (int j = 0; j < MCOLS; j++) begin
      for (int b = 0; b < 4; b++) begin
        logic [SW-1:0] t;
        t = (PROD == 0 && j % 2 == 1) ? SW'(-(D / 8 * int'(PK))) : '0;
        for (int g = 0; g < GPB; g++) t = t + SW'(s3[j][b*GPB + g]);
        u[j][b] <= t;
        v[j][b] <= SW'(u[j][b] * SW'(mr2[4*b +: 4]));
      end
      begin
        logic [SW-1:0] t;
        t = '0;
        for (int b = 0; b < 4; b++) t = t + v[j][b];
        s4[j] <= $signed(t);
      end
    end
  end
  for (genvar j = 0; j < MCOLS; j++) begin : g_col
    i2f_mid_t im;
    f32_t fi, t1, t2;
    always_ff @(posedge clk) if (en) begin
      im <= i2f_s1(32'(s4[j]));
      fi <= i2f_s2(im);
    end
    otpu_fmul #(.LAT(2)) u_m1 (.clk, .en, .a(fi), .b(ws), .y(t1));
    otpu_fmul #(.LAT(2)) u_m2 (.clk, .en, .a(t1), .b(as[j]), .y(t2));
    otpu_fadd #(.LAT(4)) u_p (.clk, .en, .a(t_even[j]), .b(t2), .y(t_pair[j]));
  end
endmodule
