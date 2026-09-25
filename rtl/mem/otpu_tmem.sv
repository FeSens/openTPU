// TMEM: fp32 scratchpad organised as LANES banks (bank = word address mod LANES). Units present
// lane-wide requests on their own ports; the slice's arbiter grants each unit all-or-nothing
// per cycle so that no bank takes more than WPB writes (and RPB reads) per cycle, and a unit
// that is not granted holds its state.
//
// Implementation (the same for simulation and the FPGA): every read port has its own copy of
// the memory, LANES banks of WORDS/LANES words, so reads never compete across ports -- except
// the guest ports (SH_MASK), rarely busy, which share one port's copy in priority order. Writes are
// broadcast to every copy: each bank takes up to WPB writes per cycle, picked from all write
// ports' lanes (WPB = 1 on the FPGA, where each bank copy is a simple dual-port BRAM with
// read-first behaviour). A read returns the old word if it is written in the same cycle.
// Each read lane's data is held until that lane reads again, so a frozen unit keeps its
// operands.
//
// Timing: the grants are the end of the slice's deepest logic (unit requests -> arbiter), and
// a write reaches every copy of its bank (NRP * the bank's block RAMs, spread over the die). So
// the selected writes are registered here and land in the block RAMs a cycle later, and a read
// of a word whose write is still in that register takes the registered data (a bypass): to the
// units the memory behaves exactly as if the write had landed in its own cycle. Likewise the
// block RAM read enables come from the requests, not the grants: a read that is not granted
// only changes a block RAM output nobody looks at (the lanes take it only the cycle after a
// granted read, and hold it otherwise).
//
// Lane routing. Every port except those in GEN_R / GEN_W (the MXU's: its drain writes scattered
// rows) carries a contiguous run: enabled lane l holds word a0 + l, where a0 is lane 0's
// address (the units drive it whenever any lane is enabled). Such a port reaches the banks
// through barrel rotators by a0 mod LANES (the lanes' bank rows are a0's row, +1 for the banks before a0's), LANES x
// log2(LANES) multiplexers instead of a LANES x LANES crossbar; the simulation stops if a
// rotator port's lanes are not contiguous. The GEN ports keep the crossbar.
//
// Within one port the enabled lanes must hit distinct banks (or read the same word); the
// simulation stops on a violation, so the timing it reports is honest.
module otpu_tmem #(
  parameter int WORDS = 1 << 16,
  parameter int LANES = 8,
  parameter int NRP   = 6,
  parameter int NWP   = 4,
  parameter int WPB   = 1,
  parameter logic [NRP-1:0] GEN_R = '1,   // read ports with the general crossbar
  parameter logic [NWP-1:0] GEN_W = '1,   // write ports with the general crossbar (WPB = 1)
  parameter int SH_HOST = 0,              // the read port whose copy SH_MASK's ports share
  parameter logic [NRP-1:0] SH_MASK = '0, // guests on SH_HOST's copy (the arbiter lets a guest
                                          // read only when no port before it on the copy asks)
  parameter int SID   = 0
) (
  input  logic                                clk,
  input  logic [NRP-1:0][LANES-1:0]           r_en,     // granted reads
  input  logic [NRP-1:0][LANES-1:0]           r_req,    // requested reads (r_en before the grant)
  input  logic [NRP-1:0][LANES-1:0][31:0]     r_addr,
  output logic [NRP-1:0][LANES-1:0][31:0]     r_data,
  input  logic [NWP-1:0][LANES-1:0]           w_en,     // granted writes
  input  logic [NWP-1:0][LANES-1:0]           w_req,    // requested writes (w_en before the grant)
  input  logic [NWP-1:0]                      w_gnt,    // the grant of each write port's unit
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
    // each port's lanes are merged per bank from its requests (a port belongs to one unit, its
    // lanes hit distinct banks), then the ports by their grants: the grant enters at the last
    // AND-OR level instead of at every lane (the arbiter admits one granted writer per bank)
    // per port, its lanes in bank order (a rotator, or the crossbar of the port's lanes, which
    // hit distinct banks), then the ports merged by their grants: the grant enters at the last
    // AND-OR level instead of at every lane (the arbiter admits one granted writer per bank)
    logic [NWP-1:0][LANES-1:0]         pv;
    logic [NWP-1:0][LANES-1:0][IW-1:0] pa;
    logic [NWP-1:0][LANES-1:0][31:0]   pdt;
    for (genvar p = 0; p < NWP; p++) begin : g_wp
      if (GEN_W[p]) begin : g_x
        always_comb begin
          pv[p] = '0; pa[p] = '0; pdt[p] = '0;
          for (int b = 0; b < LANES; b++)
            for (int l = 0; l < LANES; l++)
              if (w_req[p][l] && w_addr[p][l][BW-1:0] == BW'(b)) begin
                pv[p][b] = 1'b1;
                pa[p][b] = pa[p][b] | w_addr[p][l][BW +: IW];
                pdt[p][b] = pdt[p][b] | w_data[p][l];
              end
        end
      end else begin : g_r
        wire [BW+IW-1:0] a0 = w_addr[p][0][BW+IW-1:0];
        wire [BW-1:0]    rot = a0[BW-1:0];
        always_comb begin
          logic [LANES-1:0]       v, tv;
          logic [LANES-1:0][31:0] d, td;
          v = w_req[p]; d = w_data[p]; tv = '0; td = '0;
          for (int k = 0; k < BW; k++)
            if (rot[k]) begin
              tv = v; td = d;
              for (int l = 0; l < LANES; l++) begin
                v[(l + (1 << k)) % LANES] = tv[l];
                d[(l + (1 << k)) % LANES] = td[l];
              end
            end
          pv[p] = v; pdt[p] = d;
          for (int b = 0; b < LANES; b++) pa[p][b] = a0[BW +: IW] + IW'(BW'(b) < rot);
        end
      end
    end
    always_comb begin
      bw_v = '0; bw_a = '0; bw_d = '0;
      for (int p = 0; p < NWP; p++)
        if (w_gnt[p])
          for (int b = 0; b < LANES; b++)
            if (pv[p][b]) begin
              bw_v[b][0] = 1'b1;
              bw_a[b][0] = bw_a[b][0] | pa[p][b];
              bw_d[b][0] = bw_d[b][0] | pdt[p][b];
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

  // ---- the registered write stage: this cycle's selected writes land next cycle (pw_*); the
  // data of the writes landing now is kept one more cycle (pd) for the reads that bypass them
  logic [LANES-1:0][WPB-1:0]         pw_v;
  logic [LANES-1:0][WPB-1:0][IW-1:0] pw_a;
  logic [LANES-1:0][WPB-1:0][31:0]   pw_d, pd;
  always_ff @(posedge clk) begin
    pw_v <= bw_v;
    pw_a <= bw_a;
    pw_d <= bw_d;
    pd <= pw_d;
  end
`ifndef SYNTHESIS
  initial pw_v = '0;
`endif

  // ---- read ports: each port's bank enables and rows (from its requests)
  logic [NRP-1:0][LANES-1:0]         pb_en;
  logic [NRP-1:0][LANES-1:0][IW-1:0] pb_a;
  logic [NRP-1:0][LANES-1:0][31:0]   pq;      // the bank data each port's copy returns
  for (genvar p = 0; p < NRP; p++) begin : g_preq
    // per bank: the address of the lane that reads it (lanes of one port hit distinct banks or
    // read the same word, so an AND-OR merge is exact). A port belongs to one unit and the
    // grant is all-or-nothing per unit, so the address and the enable are merged from the
    // requests: the grant reaches no block RAM pin (it only decides which lanes take the data)
    logic [LANES-1:0]         b_en;
    logic [LANES-1:0][IW-1:0] b_a;
    wire  [BW+IW-1:0]         a0 = r_addr[p][0][BW+IW-1:0];
    wire  [BW-1:0]            rot = a0[BW-1:0];
    always_comb begin
      b_en = '0; b_a = '0;
      if (GEN_R[p]) begin
        for (int b = 0; b < LANES; b++)
          for (int l = 0; l < LANES; l++)
            if (r_addr[p][l][BW-1:0] == BW'(b) && r_req[p][l]) begin
              b_en[b] = 1'b1;
              b_a[b] = b_a[b] | r_addr[p][l][BW +: IW];
            end
      end else begin
        logic [LANES-1:0] t;
        t = '0;
        b_en = r_req[p];
        for (int k = 0; k < BW; k++)
          if (rot[k]) begin
            t = b_en;
            for (int l = 0; l < LANES; l++) b_en[(l + (1 << k)) % LANES] = t[l];
          end
        for (int b = 0; b < LANES; b++) b_a[b] = a0[BW +: IW] + IW'(BW'(b) < rot);
      end
    end
    assign pb_en[p] = b_en;
    assign pb_a[p] = b_a;
  end

  // ---- the copies: one per read port, except the guests (SH_MASK), which read SH_HOST's copy:
  // it serves the first port that requests, the host, then the guests in port order
  for (genvar p = 0; p < NRP; p++) begin : g_port
   if (!SH_MASK[p]) begin : g_copy
    logic [LANES-1:0]         b_en;
    logic [LANES-1:0][IW-1:0] b_a;
    if (p == SH_HOST && SH_MASK != '0) begin : g_sh
      always_comb begin
        logic take;
        b_en = pb_en[p]; b_a = pb_a[p];
        take = |r_req[p];
        for (int g = 0; g < NRP; g++)
          if (SH_MASK[g] && !take && |r_req[g]) begin
            b_en = pb_en[g]; b_a = pb_a[g]; take = 1'b1;
          end
      end
    end else begin : g_own
      assign b_en = pb_en[p];
      assign b_a = pb_a[p];
    end
    logic [LANES-1:0][31:0]    q, qb;
    logic [LANES-1:0][WPB-1:0] bh;      // the read hit these writes, still registered
    for (genvar b = 0; b < LANES; b++) begin : g_bank
      logic [31:0] mem [BD];
      always_ff @(posedge clk) begin
        if (b_en[b]) q[b] <= mem[b_a[b]];
        for (int w = 0; w < WPB; w++)
          if (pw_v[b][w]) mem[pw_a[b][w]] <= pw_d[b][w];
      end
      always_ff @(posedge clk)
        if (b_en[b])
          for (int w = 0; w < WPB; w++) bh[b][w] <= pw_v[b][w] && pw_a[b][w] == b_a[b];
      // the last hit wins, as in the write loop
      always_comb begin
        qb[b] = q[b];
        for (int w = 0; w < WPB; w++) if (bh[b][w]) qb[b] = pd[b][w];
      end
`ifndef SYNTHESIS
      initial for (int i = 0; i < BD; i++) mem[i] = '0;
      initial bh[b] = '0;
`endif
    end
    assign pq[p] = qb;
   end else begin : g_guest
    assign pq[p] = pq[SH_HOST];
   end
    wire  [BW-1:0]            rot = r_addr[p][0][BW-1:0];
    wire  [LANES-1:0][31:0]   qb = pq[p];
    // lane data: the bank it read (the crossbar: each lane's bank; a rotator: the run's
    // rotation), fresh the cycle after the read, then held
    logic [LANES-1:0][BW-1:0] sel;
    logic [BW-1:0]            rot_q;
    logic [LANES-1:0]         fresh;
    logic [LANES-1:0][31:0]   held, lq;
    always_ff @(posedge clk) begin
      if (|r_en[p]) rot_q <= rot;
      for (int l = 0; l < LANES; l++) begin
        fresh[l] <= r_en[p][l];
        if (GEN_R[p] && r_en[p][l]) sel[l] <= r_addr[p][l][BW-1:0];
        if (fresh[l]) held[l] <= lq[l];
      end
    end
    always_comb begin
      if (GEN_R[p]) begin
        for (int l = 0; l < LANES; l++) lq[l] = qb[sel[l]];
      end else begin
        logic [LANES-1:0][31:0] t;
        t = '0;
        lq = qb;
        for (int k = 0; k < BW; k++)
          if (rot_q[k]) begin
            t = lq;
            for (int l = 0; l < LANES; l++) lq[l] = t[(l + (1 << k)) % LANES];
          end
      end
      for (int l = 0; l < LANES; l++) r_data[p][l] = fresh[l] ? lq[l] : held[l];
    end
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
  function automatic bit contiguous(input logic [LANES-1:0] en, input logic [LANES-1:0][31:0] a);
    for (int l = 0; l < LANES; l++)
      if (en[l] && a[l] != a[0] + 32'(l)) return 0;
    return 1;
  endfunction
  always @(posedge clk) begin
    for (int p = 0; p < NRP; p++)
      if (SH_MASK[p] && |r_en[p]) begin
        if (|r_req[SH_HOST]) $fatal(1, "TMEM%0d: guest port %0d read beside its host at %0t", SID, p, $time);
        for (int g = 0; g < p; g++)
          if (SH_MASK[g] && |r_req[g]) $fatal(1, "TMEM%0d: guest port %0d read beside port %0d at %0t", SID, p, g, $time);
      end
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
      if (!GEN_R[p] && !contiguous(r_req[p], r_addr[p]))
        $fatal(1, "TMEM%0d read port %0d: lanes not contiguous at %0t", SID, p, $time);
    for (int p = 0; p < NWP; p++)
      if (WPB == 1 && !GEN_W[p] && !contiguous(w_req[p], w_addr[p]))
        $fatal(1, "TMEM%0d write port %0d: lanes not contiguous at %0t", SID, p, $time);
    for (int p = 0; p < NRP; p++)
      for (int l = 0; l < LANES; l++)
        if (r_en[p][l] && r_addr[p][l] >= WORDS) $fatal(1, "TMEM%0d read beyond %0d words", SID, WORDS);
    for (int p = 0; p < NWP; p++)
      for (int l = 0; l < LANES; l++)
        if (w_en[p][l] && w_addr[p][l] >= WORDS) $fatal(1, "TMEM%0d write beyond %0d words", SID, WORDS);
  end
  // dump: a flat shadow of the memory as the units see it (each write in its own cycle)
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
