// Lint-only stand-ins for the unisim buffers otpu_fpga_top_bcu and otpu_fpga_top_bcu_wg instantiate
// (no behaviour but BUFGCE_DIV's): `make lint` and tests/test_bcuwg_build.py check the tops
// offline; Vivado builds with the real primitives.
module IBUFDS (input logic I, input logic IB, output logic O);
  assign O = I;
endmodule

`ifndef OTPU_WG_X2                    // the x2 PHY's primitive declarations include BUFG
module BUFG (input logic I, output logic O);
  assign O = I;
endmodule
`endif

module BUFGCE (input logic I, input logic CE, output logic O);
  assign O = I & CE;
endmodule

// FDRE as the unisim model has it (INIT the configuration's value, R synchronous over CE), no
// inversions: otpu_clk_sel's (rtl/shell/otpu_ctrl.sv), the WG top's fallback clock select.
module FDRE #(parameter bit INIT = 1'b0) (input logic C, input logic CE, input logic R,
                                          input logic D, output logic Q);
  logic q = INIT;
  always_ff @(posedge C) if (R) q <= 1'b0; else if (CE) q <= D;
  assign Q = q;
endmodule

// BUFGCE_DIV as AMD's unisim model (2026.1) has it, for the WG top's clock pair and its test
// (tests/test_bcuwg_build.py's test_wg_acc_clocks_stop_and_start): the enable is taken while the
// input is low (a latch: no glitch), the divider counts the enabled input's rising edges and holds
// while it is off, CLR (asynchronous) clears the enable and the divider and holds the output low.
// At /1 the output is the enabled input; at an even divide N it starts low and toggles at every
// N/2-th enabled rising edge (configuration's GSR as time 0). +bufgce_div_o1 starts each divider
// at the other phase instead (its output high, its count at its end): a start-up that only CLR
// puts right. Only /1 and even divides, CE_TYPE SYNC, no inversions: the WG top's.
module BUFGCE_DIV #(
  parameter int    BUFGCE_DIVIDE   = 1,
  parameter string CE_TYPE         = "SYNC",
  parameter string HARDSYNC_CLR    = "FALSE",
  parameter bit    IS_CE_INVERTED  = 1'b0,
  parameter bit    IS_CLR_INVERTED = 1'b0,
  parameter bit    IS_I_INVERTED   = 1'b0,
  parameter string SIM_DEVICE      = "ULTRASCALE",
  parameter string STARTUP_SYNC    = "FALSE"
) (input logic I, input logic CE, input logic CLR, output logic O);
  initial
    if (!(BUFGCE_DIVIDE == 1 || BUFGCE_DIVIDE % 2 == 0) || CE_TYPE != "SYNC" || HARDSYNC_CLR != "FALSE"
        || IS_CE_INVERTED || IS_CLR_INVERTED || IS_I_INVERTED || STARTUP_SYNC != "FALSE"
        || SIM_DEVICE == "")
      $fatal(1, "BUFGCE_DIV stand-in: /1 or an even divide, CE_TYPE SYNC, no inversions");
  logic ce_en;
  always_latch
    if (CLR) ce_en = 1'b0;
    else if (!I) ce_en = CE;
  if (BUFGCE_DIVIDE == 1) begin : g_d1
    assign O = I && ce_en;
  end else begin : g_dn
    localparam int H = BUFGCE_DIVIDE / 2;
    logic [$clog2(H + 1)-1:0] n;
    logic o;
    initial begin
      o = $test$plusargs("bufgce_div_o1");
      n = o ? '1 : '0;
    end
    // (clocked by I with the enable: the enabled input rises only with I, the enable taken while
    // I is low; CLR a level, the divider cleared at each input edge it is high for, its output
    // low throughout)
    always @(posedge I or posedge CLR)
      if (CLR) begin n <= '0; o <= 1'b0; end
      else if (ce_en) begin
        if (n == ($clog2(H + 1))'(H - 1)) begin n <= '0; o <= !o; end
        else n <= n + 1'b1;
      end
    assign O = o && !CLR;
  end
endmodule

module IOBUF (inout wire IO, input logic I, input logic T, output logic O);
  assign IO = T ? 1'bz : I;
  assign O = IO;
endmodule

module DCIRESET (input logic RST, output logic LOCKED);
  assign LOCKED = !RST;
endmodule
