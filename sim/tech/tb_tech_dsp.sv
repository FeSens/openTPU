// The MXU's arithmetic cells (rtl/tech/<family>/otpu_tech_dsp.sv) under random stimulus: every
// cycle prints each otpu_pe variant's p and pcout (low 24 bits: all the chain uses; the packed
// variants, PK, 32) and the accumulators of otpu_mxce at one and two chains a group. Built once
// with rtl/tech/generic and once with a vendor family (and its DSP model); tests/test_tech.py
// compares the two traces (xilinx7 on yosys's DSP48E1 model); xilinxus runs on the
// unisim DSP48E2 in Vivado's simulator.
module tb_tech_dsp;
  logic clk = 1'b0;
  logic en;
  logic [7:0] act, wlo, whi;
  logic sel;
  logic [47:0] pcin, pcin2;
  logic [7:0][23:0] a, d;
  logic [7:0][16:0] bm;
  logic first, den;
  integer seed, cyc;

  // the eight otpu_pe variants FIRST, LAST, WREG, each plain and packed (PK: its chain 32 bits)
  logic [7:0][47:0] pco, pkco;
  logic signed [7:0][23:0] po;
  logic signed [7:0][31:0] pko;
  for (genvar v = 0; v < 8; v++) begin : g_pe
    otpu_pe #(.FIRST(v[0]), .LAST(v[1]), .WREG(v[2])) u (
      .clk, .en, .act, .wlo, .whi, .sel, .pcin, .pcout(pco[v]), .p(po[v]));
    otpu_pe #(.FIRST(v[0]), .LAST(v[1]), .WREG(v[2]), .PK(1'b1)) u_pk (
      .clk, .en, .act, .wlo, .whi, .sel, .pcin(pcin2), .pcout(pkco[v]), .p(pko[v]));
  end
  logic [47:0] acc1, acc2;
  otpu_mxce #(.CPG(1)) u_ce1 (.clk, .a, .d, .den, .bm, .first, .acc(acc1));
  otpu_mxce #(.CPG(2)) u_ce2 (.clk, .a, .d, .den, .bm, .first, .acc(acc2));

  // a random signed 24-bit chain sum, sign-extended (the cascade's input)
  function automatic logic [47:0] rnd48();
    logic [23:0] x;
    x = $random(seed);
    return {{24{x[23]}}, x};
  endfunction

  initial begin
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    for (cyc = 0; cyc < 4000; cyc++) begin
      en = (cyc < 16) ? 1'b1 : ($random(seed) % 8 != 0);
      act = $random(seed); wlo = $random(seed); whi = $random(seed); sel = $random(seed);
      pcin = rnd48();
      pcin2 = {{16{pcin[23]}}, pcin[23:0], 8'($random(seed))};   // a packed chain's 32-bit sum
      // chain ends within the MXU's range (|sum| < 2^19), B < 2^17, a row every 1-8 blocks
      for (int g = 0; g < 8; g++) begin
        a[g] = 24'($signed($random(seed)) % (1 << 19));
        d[g] = 24'($signed($random(seed)) % (1 << 19));
        bm[g] = $random(seed);
      end
      first = (cyc < 4) || ($random(seed) % 8 == 0);
      den = $random(seed);
      #1 clk = 1'b1;
      #1 clk = 1'b0;
      if (cyc >= 16) begin                     // the pipelines filled with defined values
        $write("%0d", cyc);
        for (int v = 0; v < 8; v++)
          $write(" %06x %06x", po[v][23:0], v[1] ? 24'd0 : pco[v][23:0]);
        for (int v = 0; v < 8; v++)
          $write(" %08x %08x", pko[v], v[1] ? 32'd0 : pkco[v][31:0]);
        $write(" %012x %012x\n", acc1, acc2);
      end
    end
    $finish;
  end
endmodule
