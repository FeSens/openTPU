// TMEM's control state from an unknown power-up, in a four-state simulation (tools/reset_audit.py
// --tmem; docs/asic.md "Power-up state"). Read as synthesis reads it (SYNTHESIS: no simulation-
// only `initial`), every flip-flop and RAM word starts at X. A testbench yosys's `sim` runs (no
// delays: its traffic from xorshift generators the reset seeds): the reset held for a few cycles
// with the inputs idle, every word written once, then random reads of the lower half that
// conflict on the RAMs (lost reads, their replays and, with three replaying ports on one RAM
// port, their ages) with writes to the upper half. lost, frz, blk_x and blk_s must never be X
// after the reset, nor any bit of a lane's word once a granted read filled it (rdx: the XOR of
// its bits). The words are not checked against a model (sim/verilator/tb_tmem.sv does that).
// The one copy with replays and true dual-port RAMs (otpu_slice TMEM_IMPL 3 RPL TDP, the
// card's), its RAMs at the core clock. NO_RST: the instance without its reset port (the RTL
// before otpu_tmem took one).
module tb_tmem_x (
  input  logic            clk,
  input  logic            rst,
  output logic            run,     // the traffic (every word written)
  output logic [4:0]      lost, frz, blk_x, blk_s,
  output logic [4:0][3:0] rdx
);
  localparam int WORDS = 256, LANES = 4, NRP = 5, NWP = 1, HR = WORDS / LANES / 2;
  // ports 1-3 replay on RAM port B with port 0 (TOPP, never replayed), port 4 on port A with
  // the writes
  localparam logic [NRP-1:0]      SIDE = 5'b10000;
  localparam logic [NRP-1:0][2:0] PRANK = {3'd4, 3'd3, 3'd2, 3'd1, 3'd0};
  logic [NRP-1:0][LANES-1:0]       r_en, r_req;
  logic [NRP-1:0][LANES-1:0][31:0] r_addr, r_data;
  logic [NWP-1:0][LANES-1:0]       w_en;
  logic [NWP-1:0]                  w_gnt;
  logic [NWP-1:0][LANES-1:0][31:0] w_addr, w_data;
  otpu_tmem #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(1), .IMPL(3), .NSB(2),
              .SBP(2), .PRANK(PRANK), .RPL(1'b1), .TOPP(0), .TDP(1'b1), .SIDE(SIDE),
              .IQ(1'b1)) dut (
    .clk, .clk_f(1'b0),
`ifndef NO_RST
    .rst,
`endif
    .r_en, .r_req, .r_addr, .r_ben('0), .r_bad('0), .r_data, .w_en, .w_req(w_en), .w_gnt,
    .w_addr, .w_data, .blk_x, .blk_s, .lost, .frz, .dump(1'b0));

  function automatic logic [31:0] xs(input logic [31:0] v);
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    return v;
  endfunction
  logic [7:0][31:0] x;
  logic [6:0]       wc;               // the writes of every word: row wc
  always_ff @(posedge clk)
    if (rst) begin
      for (int i = 0; i < 8; i++) x[i] <= 32'h9e3779b9 * (i + 1);
      wc <= '0;
      run <= 1'b0;
    end else begin
      for (int i = 0; i < 8; i++) x[i] <= xs(x[i]);
      if (!run) wc <= wc + 7'd1;
      if (wc == 7'(2 * HR - 1)) run <= 1'b1;
    end

  always_comb begin
    for (int p = 0; p < NRP; p++)
      for (int l = 0; l < LANES; l++) begin
        // a port's lanes in distinct banks (a rotation); a replaying port holds
        r_req[p][l] = run && !frz[p] && x[p][2 + 2 * l +: 2] != 2'd0;
        r_addr[p][l] = 32'(x[p][10 + 5 * l +: 5]) * LANES + 32'((l + int'(x[p][1:0])) % LANES);
        r_en[p][l] = r_req[p][l] && x[p][31:30] != 2'd0;
      end
    w_gnt[0] = !run || x[5][0];
    for (int l = 0; l < LANES; l++) begin
      w_en[0][l] = !run || (w_gnt[0] && x[5][1 + 2 * l +: 2] != 2'd0);
      w_addr[0][l] = run ? 32'(HR + x[6][5 * l +: 5]) * LANES
                           + 32'((l + int'(x[5][11:10])) % LANES)
                         : 32'(wc) * LANES + 32'(l);
      w_data[0][l] = x[7] ^ 32'(l);
    end
  end

  logic [NRP-1:0][LANES-1:0] filled;  // the lane holds a granted read's word
  always_ff @(posedge clk)
    for (int p = 0; p < NRP; p++)
      filled[p] <= rst ? '0 : filled[p] | (lost[p] ? '0 : r_en[p]);
  always_comb
    for (int p = 0; p < NRP; p++)
      for (int l = 0; l < LANES; l++) rdx[p][l] = filled[p][l] && ^r_data[p][l];
endmodule
