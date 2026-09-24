// TMEM: fp32 scratchpad organised as LANES banks (bank = word address mod LANES). Units present
// lane-wide requests on their own ports; the slice's arbiter grants each unit all-or-nothing
// per cycle so that no bank takes more than WPB writes (and RPB reads) per cycle, and a unit
// that is not granted holds its state.
//
// Implementation (the same for simulation and the FPGA): every read port has its own copy of
// the memory, LANES banks of WORDS/LANES words, so reads never compete across ports. Writes are
// broadcast to every copy: each bank takes up to WPB writes per cycle, picked from all write
// ports' lanes (WPB = 1 on the FPGA, where each bank copy is a simple dual-port BRAM with
// read-first behaviour). A read returns the old word if it is written in the same cycle.
// Each read lane's data is held until that lane reads again, so a frozen unit keeps its
// operands.
//
// Within one port the enabled lanes must hit distinct banks (or read the same word); the
// simulation stops on a violation, so the timing it reports is honest.
module otpu_tmem #(
  parameter int WORDS = 1 << 16,
  parameter int LANES = 8,
  parameter int NRP   = 6,
  parameter int NWP   = 4,
  parameter int WPB   = 1,
  parameter int SID   = 0
) (
  input  logic                                clk,
  input  logic [NRP-1:0][LANES-1:0]           r_en,     // granted reads
  input  logic [NRP-1:0][LANES-1:0]           r_req,    // requested reads (r_en before the grant)
  input  logic [NRP-1:0][LANES-1:0][31:0]     r_addr,
  output logic [NRP-1:0][LANES-1:0][31:0]     r_data,
  input  logic [NWP-1:0][LANES-1:0]           w_en,
  input  logic [NWP-1:0][LANES-1:0][31:0]     w_addr,
  input  logic [NWP-1:0][LANES-1:0][31:0]     w_data,
  input  logic                                dump
);
  localparam int BW = $clog2(LANES);
  localparam int BD = WORDS / LANES;        // words per bank
  localparam int IW = $clog2(BD);

  // ---- write selection: per bank, the enabled lanes of all write ports. The arbiter admits at
  // most WPB of them, so with WPB = 1 a flat AND-OR tree selects the one (no priority chain).
  logic [LANES-1:0][WPB-1:0]         bw_v;
  logic [LANES-1:0][WPB-1:0][IW-1:0] bw_a;
  logic [LANES-1:0][WPB-1:0][31:0]   bw_d;
  if (WPB == 1) begin : g_w1
    always_comb begin
      bw_v = '0; bw_a = '0; bw_d = '0;
      for (int b = 0; b < LANES; b++)
        for (int p = 0; p < NWP; p++)
          for (int l = 0; l < LANES; l++)
            if (w_en[p][l] && w_addr[p][l][BW-1:0] == BW'(b)) begin
              bw_v[b][0] = 1'b1;
              bw_a[b][0] = bw_a[b][0] | w_addr[p][l][BW +: IW];
              bw_d[b][0] = bw_d[b][0] | w_data[p][l];
            end
    end
  end else begin : g_wn
    always_comb begin
      for (int b = 0; b < LANES; b++) begin
        int n;
        n = 0;
        bw_v[b] = '0; bw_a[b] = '0; bw_d[b] = '0;
        for (int p = 0; p < NWP; p++)
          for (int l = 0; l < LANES; l++)
            if (w_en[p][l] && w_addr[p][l][BW-1:0] == BW'(b) && n < WPB) begin
              bw_v[b][n] = 1'b1;
              bw_a[b][n] = w_addr[p][l][BW +: IW];
              bw_d[b][n] = w_data[p][l];
              n = n + 1;
            end
      end
    end
  end

  // ---- read ports
  for (genvar p = 0; p < NRP; p++) begin : g_port
    // per bank: the address of the lane that reads it (lanes of one port hit distinct banks or
    // read the same word, so an AND-OR merge is exact). A port belongs to one unit and the
    // grant is all-or-nothing per unit, so the address is merged from the requests and the
    // grant only reaches the block RAM's enable, not its address pins.
    logic [LANES-1:0]         b_en;
    logic [LANES-1:0][IW-1:0] b_a;
    always_comb begin
      b_en = '0; b_a = '0;
      for (int b = 0; b < LANES; b++)
        for (int l = 0; l < LANES; l++)
          if (r_addr[p][l][BW-1:0] == BW'(b)) begin
            if (r_en[p][l]) b_en[b] = 1'b1;
            if (r_req[p][l]) b_a[b] = b_a[b] | r_addr[p][l][BW +: IW];
          end
    end
    logic [LANES-1:0][31:0] q;
    for (genvar b = 0; b < LANES; b++) begin : g_bank
      logic [31:0] mem [BD];
      always_ff @(posedge clk) begin
        if (b_en[b]) q[b] <= mem[b_a[b]];
        for (int w = 0; w < WPB; w++)
          if (bw_v[b][w]) mem[bw_a[b][w]] <= bw_d[b][w];
      end
`ifndef SYNTHESIS
      initial for (int i = 0; i < BD; i++) mem[i] = '0;
`endif
    end
    // lane data: the bank it read, fresh the cycle after the read, then held
    logic [LANES-1:0][BW-1:0] sel;
    logic [LANES-1:0]         fresh;
    logic [LANES-1:0][31:0]   held;
    always_ff @(posedge clk) begin
      for (int l = 0; l < LANES; l++) begin
        fresh[l] <= r_en[p][l];
        if (r_en[p][l]) sel[l] <= r_addr[p][l][BW-1:0];
        if (fresh[l]) held[l] <= q[sel[l]];
      end
    end
    always_comb
      for (int l = 0; l < LANES; l++) r_data[p][l] = fresh[l] ? q[sel[l]] : held[l];
  end

`ifndef SYNTHESIS
  function automatic bit conflict(input logic [LANES-1:0] en, input logic [LANES-1:0][31:0] a,
                                  input bit is_write);
    for (int i = 0; i < LANES; i++)
      for (int j = i + 1; j < LANES; j++)
        if (en[i] && en[j] && (a[i] % LANES) == (a[j] % LANES) && (is_write || a[i] != a[j]))
          return 1;
    return 0;
  endfunction
  always @(posedge clk) begin
    for (int p = 0; p < NRP; p++)
      if (conflict(r_en[p], r_addr[p], 0)) $fatal(1, "TMEM%0d read port %0d bank conflict at %0t", SID, p, $time);
    for (int p = 0; p < NWP; p++)
      if (conflict(w_en[p], w_addr[p], 1)) $fatal(1, "TMEM%0d write port %0d bank conflict at %0t", SID, p, $time);
    for (int b = 0; b < LANES; b++) begin
      int n;
      n = 0;
      for (int p = 0; p < NWP; p++)
        for (int i = 0; i < LANES; i++)
          if (w_en[p][i] && w_addr[p][i] % LANES == b) n++;
      if (n > WPB) $fatal(1, "TMEM%0d: %0d writes to bank %0d at %0t", SID, n, b, $time);
    end
    for (int p = 0; p < NRP; p++)
      for (int l = 0; l < LANES; l++)
        if (r_en[p][l] && r_addr[p][l] >= WORDS) $fatal(1, "TMEM%0d read beyond %0d words", SID, WORDS);
    for (int p = 0; p < NWP; p++)
      for (int l = 0; l < LANES; l++)
        if (w_en[p][l] && w_addr[p][l] >= WORDS) $fatal(1, "TMEM%0d write beyond %0d words", SID, WORDS);
  end
  // dump: a flat shadow of the memory, written with the same selection as the banks
  logic [31:0] shadow [WORDS];
  initial for (int i = 0; i < WORDS; i++) shadow[i] = '0;
  always @(posedge clk)
    for (int b = 0; b < LANES; b++)
      for (int w = 0; w < WPB; w++)
        if (bw_v[b][w]) shadow[{bw_a[b][w], BW'(b)}] <= bw_d[b][w];
  string dir;
  initial void'($value$plusargs("dir=%s", dir));
  always @(posedge clk) if (dump) begin
    int fd;
    fd = $fopen($sformatf("%s/tmem_%0d.hex", dir, SID), "w");
    for (int i = 0; i < WORDS; i++) $fwrite(fd, "%08x\n", shadow[i]);
    $fclose(fd);
  end
`endif
endmodule
