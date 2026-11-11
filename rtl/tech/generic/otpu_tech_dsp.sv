// The systolic MXU's arithmetic cells (otpu_mxu IMPL 2), technology-independent behavioural
// models: the simulators' (Verilator), the ASIC flows' and any part without the FPGA families'
// DSP cells. rtl/tech/xilinx7 (DSP48E1) and rtl/tech/xilinxus (DSP48E2) implement the same
// modules, cycle for cycle; the file list picks one (opentpu/target.py, docs/targets.md).

// One position of the systolic MXU (IMPL 2): the activation registered (as a DSP's B register),
// the two weight streams lo / hi, one picked by the registered select sel, the product registered
// (M) and the chain's running sum P = PCIN + M (FIRST: P = M), registered unless LAST (the column
// end registers a chain's last sum). WREG registers the weights too (wlo, whi are their next
// values); sel is the select's next value. The pipeline is the DSP48's (rtl/tech/xilinx7 says
// why each register is where it is).
// PK (otpu_mxu at WG > 1, docs/weight_groups.md): the packed position. sel then picks
// A alone (0) or the pre-adder's A + D with D = whi * 2^16 (1): the product is
// act * wlo + act * whi * 2^16, two rows' products against one activation in one multiplier,
// and the chain's sum (p, 32 bits) holds both (the column end splits them). The fabric picks the
// low weight in place of sel's 0 / 1 (otpu_mxu). Under sel a chain's FIRST position adds 2^15
// (the DSP's RND): with both halves' sums within +-2^15, the column end splits a sum by its bits,
// p[15:0] - 2^15 and p >> 16.
module otpu_pe #(parameter bit FIRST = 1'b0, parameter bit LAST = 1'b0, parameter bit WREG = 1'b0,
                 parameter bit PK = 1'b0) (
  input  logic               clk,
  input  logic               en,
  input  logic [7:0]         act,
  input  logic [7:0]         wlo,               // WREG: the weights from the next cycle on
  input  logic [7:0]         whi,
  input  logic               sel,               // the select from the next cycle on
  input  logic [47:0]        pcin,
  output logic [47:0]        pcout,
  output logic signed [(PK ? 32 : 24)-1:0] p
);
  localparam int PB = PK ? 32 : 24, MB = PK ? 32 : 16;
  logic [7:0] br, wlr, whr;
  logic       selr;
  logic signed [MB-1:0] m;
  wire  signed [PB-1:0] pn = (FIRST ? ((PK && selr) ? PB'(32768) : PB'(0)) : $signed(pcin[PB-1:0]))
                            + PB'(m);
  wire  [7:0] wl = WREG ? wlr : wlo, wh = WREG ? whr : whi;
  always_ff @(posedge clk) if (en) begin
    br <= act;
    selr <= sel;
    wlr <= wlo;
    whr <= whi;
    if (PK) m <= MB'(int'($signed(br)) * (int'($signed(wl)) + (selr ? int'($signed(wh)) * 65536 : 0)));
    else    m <= MB'(int'($signed(br)) * int'($signed(selr ? wh : wl)));
  end
  if (LAST) begin : g_comb
    assign p = pn;
  end else begin : g_preg
    always_ff @(posedge clk) if (en) p <= pn;
  end
  assign pcout = LAST ? 48'd0 : 48'(p);              // (a chain's last cascades nowhere)
endmodule


// The MX column end (otpu_mxu, docs/mx_mxu.md): a column's eight group sums, each one chain end
// (a[g], the LAST position's unregistered sum) or, at two chains a group (CPG 2), two (a[g] +
// d[g]; den 0 drops d, as the DSP's INMODE), times the group's B = m << sh (bm[g], unsigned,
// < 2^17), summed and accumulated over the row's blocks into one 48-bit integer:
//   acc = (first ? 0 : acc) + sum_g (a[g] + d[g]) * bm[g]
// Edges from E, the one that registers a / d / bm: groups 0..3 multiply at E+1 and are summed at
// E+2 (p3); groups 4..7 register their pre-adder's sum and B again at E+1, multiply at E+2 and
// join p3 at E+3 (p7); the accumulator adds p7 at E+4. first is registered at E+3 (two cycles
// after its block's a / d / bm), as the accumulator's OPMODE. Exact: |a + d| < 2^24, the
// products < 2^41 and a row's sum < 2^47 (docs/isa.md MM). rtl/tech/xilinx7 maps it onto nine
// DSP48E1s (eight multipliers on two cascades, one accumulator).
module otpu_mxce #(parameter int CPG = 1) (
  input  logic                clk,
  input  logic [7:0][23:0]    a,        // group g's chain end: the pre-adder's A ...
  input  logic [7:0][23:0]    d,        // ... and D (CPG 2: the group's second chain, or
                                        // otpu_mxu's packed int8 high halves; else unused)
  input  logic                den,      // D into the pre-adder (CPG 2; constant while it streams)
  input  logic [7:0][16:0]    bm,       // B = m << sh, with a / d
  input  logic                first,    // the row's first block (two cycles after a / d / bm)
  output logic [47:0]         acc
);
  initial if (CPG != 1 && CPG != 2) $fatal(1, "otpu_mxce: CPG %0d", CPG);
  logic [7:0][23:0] ar, dr;
  logic [7:0][16:0] br;
  logic [7:4][24:0] adr;                   // groups 4..7: the pre-adder's register (ADREG) ...
  logic [7:4][16:0] br2;                   // ... and B's second register
  logic [7:0][47:0] m;
  logic [47:0]      p3, p7;
  logic             fr;
  // 25 bits, as the DSP48E1's pre-adder
  function automatic logic [24:0] pre(input logic [23:0] x, input logic [23:0] y);
    return {x[23], x} + {y[23], y};
  endfunction
  always_ff @(posedge clk) begin
    ar <= a; dr <= (CPG == 2) ? d : '0; br <= bm;
    for (int g = 0; g < 4; g++)
      m[g] <= 48'($signed(pre(ar[g], den ? dr[g] : 24'd0))) * 48'($signed({1'b0, br[g]}));
    for (int g = 4; g < 8; g++) begin
      adr[g] <= pre(ar[g], den ? dr[g] : 24'd0);
      br2[g] <= br[g];
      m[g] <= 48'($signed(adr[g])) * 48'($signed({1'b0, br2[g]}));
    end
    p3 <= m[0] + m[1] + m[2] + m[3];
    p7 <= p3 + m[4] + m[5] + m[6] + m[7];
    fr <= first;
    acc <= (fr ? 48'd0 : acc) + p7;
  end
endmodule


// A position of the pumped packed column (otpu_mxu PUMP, otpu_mxpc): one DSP48E2's registers at
// clk_f = 2 x clk. A2 = wlo and D = whi * 2^16 (loaded with ce_w: the slow edges), AD = A2 + D
// (loaded with ce_ad, held two fast cycles), B = the activation (BF: one register, the fabric's
// b_f in front of it; else two, B1 and B2), M = B * AD and P = PCIN + M (FIRST: P = M + 2^15,
// the DSP's RND), every register on: rtl/tech/xilinxus maps it onto one DSP48E2.
module otpu_ppe #(parameter bit FIRST = 1'b0, parameter bit BF = 1'b1) (
  input  logic        clk_f,
  input  logic        ce_w,
  input  logic        ce_ad,
  input  logic [7:0]  wlo,
  input  logic [7:0]  whi,
  input  logic [7:0]  act,
  input  logic [47:0] pcin,
  output logic [47:0] pcout,
  output logic [31:0] p
);
  logic signed [7:0]  a2, d2, b1, b2;
  logic signed [26:0] ad;
  logic signed [44:0] m;
  logic [47:0]        pr;
  wire  signed [7:0]  b = BF ? b1 : b2;
  always_ff @(posedge clk_f) begin
    if (ce_w) begin
      a2 <= wlo;
      d2 <= whi;
    end
    if (ce_ad) ad <= 27'(a2) + 27'(d2) * 27'sd65536;
    b1 <= act;
    b2 <= b1;
    m <= 45'(b) * 45'(ad);
    pr <= (FIRST ? 48'h8000 : pcin) + 48'(m);
  end
  assign pcout = pr;
  assign p = pr[31:0];
endmodule
