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
//
// IMPL 1 (banked; an ASIC, whose SRAM macros cannot be replicated per read port at no cost):
// NCP copies instead of one per read port, each LANES banks of one 1R1W RAM (otpu_ram_1r1w, the
// target's technology cell). CPY[p] is read port p's copy; ports of one unit (the slice's
// arbitration group) sit on different copies, so a unit alone always fits. Each copy's bank
// serves one read per cycle: the slice's arbiter admits at most one port per copy and bank (a
// port's lanes in one bank read one word), and the bank's address comes from that port's granted
// lanes (r_en), not its requests. Writes are broadcast to every copy, one per bank (WPB = 1).
// The registered write stage, the bypass and the lanes' data path are IMPL 0's.
// RQ (IMPL 1): a copy's bank address comes from the requests instead, the port first in the
// slice's priority order (PRANK, smaller first) among those on the copy that ask for the bank;
// the slice's arbiter then grants a unit only where it is that port (otpu_slice TMEM_RQ), so the
// granted port's address is the one at the RAM and the grants reach no RAM pin (on the FPGA, the
// grant-to-address path of IMPL 1 does not close at the core clock: docs/tmem_pump.md).
module otpu_tmem #(
  parameter int WORDS = 1 << 16,
  parameter int LANES = 8,
  parameter int NRP   = 6,
  parameter int NWP   = 4,
  parameter int WPB   = 1,
  parameter logic [NRP-1:0] GEN_R = '1,   // read ports with the general crossbar
  parameter logic [NRP-1:0] PBR   = '0,   // GEN_R ports whose unit merges its lanes by bank
                                          //   (r_ben / r_bad: from its flip-flops, see g_preq)
  parameter logic [NWP-1:0] GEN_W = '1,   // write ports with the general crossbar (WPB 1, IMPL 3's 2)
  parameter int SH_HOST = 0,              // the read port whose copy SH_MASK's ports share
  parameter logic [NRP-1:0] SH_MASK = '0, // guests on SH_HOST's copy (the arbiter lets a guest
                                          // read only when no port before it on the copy asks)
  parameter int SID   = 0,
  parameter int IMPL  = 0,                // 0: a copy per read port (block RAM); 1: banked (see top)
  parameter int NCP   = 1,                // IMPL 1: copies
  parameter logic [NRP-1:0][2:0] CPY = '0, // IMPL 1: each read port's copy (< NCP)
  parameter bit   RQ    = 1'b0,           // IMPL 1: bank addresses from the requests (see top)
  parameter logic [NRP-1:0][2:0] PRANK = '0, // RQ, IMPL 3: each read port's priority (0 first)
  parameter int   NSB   = 8,              // IMPL 3: RAMs (sub-banks) per bank
  parameter int   SBP   = 3,              // IMPL 3: the sub-bank field's lowest word-address bit
  parameter bit   SHARE = 1'b1,           // IMPL 3: ports reading one word share its RAM read
  parameter logic [NRP-1:0][2:0] GRP = '0, // IMPL 3: each read port's group (the slice's unit)
  parameter int   SBH   = 0,              // IMPL 3: the sub-bank hashed with the row bits above
                                          //   it (1: the next field, 2: every field, XORed)
  parameter bit   RPL   = 1'b0,           // IMPL 3: a lost read is replayed the next cycle (see
                                          //   IMPL 3 below)
  parameter logic [NRP-1:0] RPLM = '1,    // RPL: the ports whose reads may be replayed
  parameter int   TOPP  = 0,              // RPL: the port whose request comes before the replays
  parameter bit   TDP   = 1'b0,           // IMPL 3: true dual-port RAMs (see IMPL 3 below)
  parameter logic [NRP-1:0] SIDE = '0,    // TDP: each read port's RAM port (1: A, the writes')
  parameter bit   GST   = 1'b0,           // IMPL 3: SH_MASK's ports are SH_HOST's guests (see IMPL 3)
  parameter int   PMP   = 0,              // TDP: 1 the RAMs pumped at 2x on clk_f (otpu_pump_bank), 2
                                          //   its model at the core clock (see IMPL 3)
  parameter bit   IQ    = 1'b0,           // the input stage: the granted writes registered (below)
  parameter string STYLE = "block"        // the pumped RAMs' (PMP 1: otpu_pump_bank STYLE; "ultra":
                                          //   UltraRAM, the WG targets': its block RAM is the board's; the
                                          //   other configurations' RAMs are block RAM)
) (
  input  logic                                clk,
  input  logic                                clk_f,    // PMP 1: 2 x clk, aligned (else unused)
  input  logic                                rst,      // synchronous: the control state below
  input  logic [NRP-1:0][LANES-1:0]           r_en,     // granted reads
  input  logic [NRP-1:0][LANES-1:0]           r_req,    // requested reads (r_en before the grant)
  input  logic [NRP-1:0][LANES-1:0][31:0]     r_addr,
  input  logic [NRP-1:0][LANES-1:0]           r_ben,    // PBR: r_req by bank (bank b's)
  input  logic [NRP-1:0][LANES-1:0][31:0]     r_bad,    // PBR: the address of bank b's lane
  output logic [NRP-1:0][LANES-1:0][31:0]     r_data,
  input  logic [NWP-1:0][LANES-1:0]           w_en,     // granted writes
  input  logic [NWP-1:0][LANES-1:0]           w_req,    // requested writes (w_en before the grant)
  input  logic [NWP-1:0]                      w_gnt,    // the grant of each write port's unit
  input  logic [NWP-1:0][LANES-1:0][31:0]     w_addr,
  input  logic [NWP-1:0][LANES-1:0][31:0]     w_data,
  output logic [NRP-1:0]                      blk_x,    // IMPL 3: a port of another group
  output logic [NRP-1:0]                      blk_s,    //   / of its own took a RAM it reads
  output logic [NRP-1:0]                      lost,     // IMPL 3 RPL: the port's read this cycle
                                                        //   lost a RAM (replayed next cycle)
  output logic [NRP-1:0]                      frz,      // IMPL 3 RPL: the port replays now
  input  logic                                dump
);
  localparam int BW = $clog2(LANES);
  localparam int BD = WORDS / LANES;        // words per bank
  localparam int IW = $clog2(BD);
  initial if (IMPL == 1 && WPB != 1) $fatal(1, "otpu_tmem: IMPL 1 needs WPB = 1");
  initial if (IMPL == 3 && !(WPB == 1 || (WPB == 2 && PMP != 0)))
    $fatal(1, "otpu_tmem: IMPL 3 takes WPB 1, or 2 pumped (PMP 1 / 2: the odd edge's two ports)");
  initial if (TDP && IMPL != 3) $fatal(1, "otpu_tmem: TDP needs IMPL 3");
  initial if (GST && (IMPL != 3 || (RPL && SH_HOST != TOPP)))
    $fatal(1, "otpu_tmem: GST needs IMPL 3, and SH_HOST = TOPP with RPL (no replay of a guest's read)");
  initial if (IMPL == 3 && (NSB < 1 || (1 << $clog2(NSB)) != NSB || SBP < BW || SBP + $clog2(NSB) > BW + IW))
    $fatal(1, "otpu_tmem: IMPL 3 sub-banks NSB %0d (a power of 2) at bit SBP %0d", NSB, SBP);
  initial if (PMP != 0 && !TDP) $fatal(1, "otpu_tmem: PMP needs TDP (its read ports' sides)");
  initial if (STYLE != "block" && STYLE != "ultra") $fatal(1, "otpu_tmem: STYLE %s", STYLE);
  initial if (IMPL == 1) for (int p = 0; p < NRP; p++)
    if (CPY[p] >= NCP) $fatal(1, "otpu_tmem: port %0d's copy %0d >= NCP %0d", p, CPY[p], NCP);
  // the read port that holds each port's copy: IMPL 0, the port itself or SH_HOST (a guest);
  // IMPL 1, the first port on its copy
  function automatic int host_of(input int p);
    if (IMPL == 0) return SH_MASK[p] ? SH_HOST : p;
    for (int q = 0; q < NRP; q++) if (CPY[q] == CPY[p]) return q;
    return p;
  endfunction
`ifndef SYNTHESIS
  // the checks start once the reset has held two edges (registers start arbitrary, and so does a
  // unit's reset register until the slice's reset pipeline fills: otpu_slice)
  bit rst_p, rst_seen;
  initial begin rst_p = 1'b0; rst_seen = 1'b0; end
  always @(posedge clk) begin rst_p <= rst; if (rst && rst_p) rst_seen <= 1'b1; end
  wire chk = rst_seen && !rst;
`endif

  // ---- IQ, the input stage (release r2, docs/tmem_nocopy.md 7): the granted writes (their
  // lanes, addresses and data, the ports' grants) are registered here, and the lane routing, the
  // write selection and the registered write stage run on them a cycle later, so the selection
  // starts at flip-flops instead of at the units' write logic and the arbiter's grants. The units
  // register their read requests (the reads come a cycle after they form, with the same grants),
  // so reads and writes both move a cycle and the memory's order is unchanged: no bypass of the
  // stage. Without IQ the inputs pass through.
  logic [NWP-1:0][LANES-1:0]       wi_en;
  logic [NWP-1:0]                  wi_gnt;
  logic [NWP-1:0][LANES-1:0][31:0] wi_addr, wi_data;
  if (IQ) begin : g_iq
    always_ff @(posedge clk) begin
      wi_en <= rst ? '0 : w_en;
      wi_gnt <= rst ? '0 : w_gnt;
      // with WPB 1, and IMPL 3's WPB 2, every cycle (their selection, g_w1, takes a port's lanes
      // only under its registered grant, wi_gnt, so the words are the granted ones there): the
      // grant, the arbiter's logic, then enables no address and data flip-flops (on the m8 card
      // one LUT of it enabled 11,254, the MXU's lanes with them; WPB 2's acf2 arbiter is deeper)
      for (int p = 0; p < NWP; p++)
        if (w_gnt[p] || WPB == 1 || IMPL == 3) begin
          wi_addr[p] <= w_addr[p];
          wi_data[p] <= w_data[p];
        end
    end
  end else begin : g_niq
    assign wi_en = w_en, wi_gnt = w_gnt, wi_addr = w_addr, wi_data = w_data;
  end
  // (the selection takes each port's lanes from its requests and the port by its grant: with IQ
  // the registered granted lanes and grants)
  wire [NWP-1:0][LANES-1:0] wi_req = IQ ? wi_en : w_req;

  // ---- write selection: per bank, the enabled lanes of all write ports. The arbiter admits at
  // most WPB of them, so with WPB = 1 a flat AND-OR tree selects the one (no priority chain);
  // IMPL 3's WPB 2 takes the same per-port merge, then per bank the first granted port and the
  // last (when they are two)
  logic [LANES-1:0][WPB-1:0]         bw_v;
  logic [LANES-1:0][WPB-1:0][IW-1:0] bw_a;
  logic [LANES-1:0][WPB-1:0][31:0]   bw_d;
  if (WPB == 1 || IMPL == 3) begin : g_w1
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
              if (wi_req[p][l] && wi_addr[p][l][BW-1:0] == BW'(b)) begin
                pv[p][b] = 1'b1;
                pa[p][b] = pa[p][b] | wi_addr[p][l][BW +: IW];
                pdt[p][b] = pdt[p][b] | wi_data[p][l];
              end
        end
      end else begin : g_r
        wire [BW+IW-1:0] a0 = wi_addr[p][0][BW+IW-1:0];
        wire [BW-1:0]    rot = a0[BW-1:0];
        always_comb begin
          logic [LANES-1:0]       v, tv;
          logic [LANES-1:0][31:0] d, td;
          v = wi_req[p]; d = wi_data[p]; tv = '0; td = '0;
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
    if (WPB == 1) begin : g_m1
      always_comb begin
        bw_v = '0; bw_a = '0; bw_d = '0;
        for (int p = 0; p < NWP; p++)
          if (wi_gnt[p])
            for (int b = 0; b < LANES; b++)
              if (pv[p][b]) begin
                bw_v[b][0] = 1'b1;
                bw_a[b][0] = bw_a[b][0] | pa[p][b];
                bw_d[b][0] = bw_d[b][0] | pdt[p][b];
              end
      end
    end else begin : g_m2
      // per bank the granted writers (at most two, the arbiter's), one-hot first and last: the
      // first takes slot 0, the last slot 1 when it is another
      always_comb
        for (int b = 0; b < LANES; b++) begin
          logic [NWP-1:0] v, fh, lh;
          for (int q = 0; q < NWP; q++) v[q] = wi_gnt[q] && pv[q][b];
          for (int q = 0; q < NWP; q++) begin
            fh[q] = v[q];
            lh[q] = v[q];
            for (int o = 0; o < q; o++) if (v[o]) fh[q] = 1'b0;
            for (int o = q + 1; o < NWP; o++) if (v[o]) lh[q] = 1'b0;
          end
          bw_v[b] = '0; bw_a[b] = '0; bw_d[b] = '0;
          for (int q = 0; q < NWP; q++) begin
            if (fh[q]) begin
              bw_v[b][0] = 1'b1;
              bw_a[b][0] = bw_a[b][0] | pa[q][b];
              bw_d[b][0] = bw_d[b][0] | pdt[q][b];
            end
            if (lh[q] && !fh[q]) begin
              bw_v[b][WPB-1] = 1'b1;
              bw_a[b][WPB-1] = bw_a[b][WPB-1] | pa[q][b];
              bw_d[b][WPB-1] = bw_d[b][WPB-1] | pdt[q][b];
            end
          end
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
            if (wi_en[p][l] && wi_addr[p][l][BW-1:0] == BW'(b) && n < WPB) begin
              bw_v[b][n] = 1'b1;
              bw_a[b][n] = wi_addr[p][l][BW +: IW];
              bw_d[b][n] = wi_data[p][l];
              n = n + 1;
            end
      end
    end
  end

  // ---- the registered write stage: this cycle's selected writes land next cycle (pw_*); the
  // data of the writes landing now is kept one more cycle (pd) for the reads that bypass them.
  // Each pw bit reaches every copy of its bank (the copies' block RAMs spread over the die):
  // synthesis replicates the registers so each copy gets nearby ones (one copy's net to the
  // block RAM address pins was the worst core_clk path at 120 MHz, 0 logic levels). With PMP 1
  // each pumped RAM takes its own copy instead (pk_*, IMPL 3); pw stays for the bypass, the
  // replays' checks and the dump.
  (* max_fanout = 16 *) logic [LANES-1:0][WPB-1:0]         pw_v;
  (* max_fanout = 16 *) logic [LANES-1:0][WPB-1:0][IW-1:0] pw_a;
  (* max_fanout = 16 *) logic [LANES-1:0][WPB-1:0][31:0]   pw_d;
  logic [LANES-1:0][WPB-1:0][31:0]                         pd;
  always_ff @(posedge clk) begin
    pw_v <= rst ? '0 : bw_v;
    pw_a <= bw_a;
    pw_d <= bw_d;
    pd <= pw_d;
  end

  // ---- read ports: each port's bank enables and rows (from its requests)
  logic [NRP-1:0][LANES-1:0]         pb_en;
  logic [NRP-1:0][LANES-1:0][IW-1:0] pb_a;
  logic [NRP-1:0][LANES-1:0][31:0]   pq;      // the bank data each port's copy returns
  // IMPL 3 without the bypass (PMP 1), a host port (pdir): its side's RAM words, bank b's RAM k at
  // b * NSB + k (pr), and the RAM each bank read (pss); its lanes select a word by {bank, RAM}, one
  // NSB x LANES : 1 select from the RAMs (LUT6s, MUXF7s, a MUXF8 in a slice at NSB 2), not the
  // bank's RAM select and then the lane's bank select (two LUTs and a route between: the m8 card
  // at 150 MHz, the pumped RAMs -> 4 levels -> the VPU's, the quantizer's, the DMA's capture,
  // -0.112 ns of a full core cycle, 3.9 ns of route)
  localparam int PSB = (NSB > 1) ? $clog2(NSB) : 1;
  logic [NRP-1:0][LANES*NSB-1:0][31:0] pr;
  logic [NRP-1:0][LANES-1:0][PSB-1:0]  pss;
  logic [NRP-1:0]                      pdir;
  logic [NRP-1:0][LANES-1:0]         rpf;     // IMPL 3 RPL: the lanes a replay reads now
  for (genvar p = 0; p < NRP; p++) begin : g_preq
    // per bank: the address of the lane that reads it (lanes of one port hit distinct banks or
    // read the same word, so an AND-OR merge is exact). A port belongs to one unit and the
    // grant is all-or-nothing per unit, so the address and the enable are merged from the
    // requests: the grant reaches no block RAM pin (it only decides which lanes take the data)
    logic [LANES-1:0]         b_en;
    logic [LANES-1:0][IW-1:0] b_a;
    wire  [BW+IW-1:0]         a0 = r_addr[p][0][BW+IW-1:0];
    wire  [BW-1:0]            rot = a0[BW-1:0];
    // IMPL 1: a copy's bank serves the one port granted in it, so its enables are the grants
    // (RQ: the requests, the copy picks by priority)
    wire  [LANES-1:0]         rs = (IMPL == 1 && !RQ) ? r_en[p] : r_req[p];
    // the port's own merge: the crossbar's (GEN_R: per bank, the lanes that read it), else the
    // rotator's (lanes contiguous from lane 0's word: bank b is lane (b - rot)'s, one row on
    // where the lanes wrap)
    logic [LANES-1:0]         x_en;
    logic [LANES-1:0][IW-1:0] x_a;
    always_comb begin
      x_en = '0; x_a = '0;
      if (GEN_R[p]) begin
        for (int b = 0; b < LANES; b++)
          for (int l = 0; l < LANES; l++)
            if (r_addr[p][l][BW-1:0] == BW'(b) && rs[l]) begin
              x_en[b] = 1'b1;
              x_a[b] = x_a[b] | r_addr[p][l][BW +: IW];
            end
      end else begin
        logic [LANES-1:0] t;
        t = '0;
        x_en = rs;
        for (int k = 0; k < BW; k++)
          if (rot[k]) begin
            t = x_en;
            for (int l = 0; l < LANES; l++) x_en[(l + (1 << k)) % LANES] = t[l];
          end
        for (int b = 0; b < LANES; b++) x_a[b] = a0[BW +: IW] + IW'(BW'(b) < rot);
      end
    end
    // PBR: the unit's merge, from its flip-flops (otpu_mxu TQ: rb_q / rb_a; otpu_tmem_breq's
    // decode registered in the DMA, the VPU and the quantizer), without the lanes' merge in front
    // of the ranks, the compares and the RAM pins
    always_comb begin
      b_en = x_en; b_a = x_a;
      if (PBR[p]) begin
        b_en = r_ben[p];
        for (int b = 0; b < LANES; b++) b_a[b] = r_bad[p][b][BW +: IW];
      end
    end
    assign pb_en[p] = b_en;
    assign pb_a[p] = b_a;
`ifndef SYNTHESIS
    // PBR: the unit's merge is the port's own (the banks read; their rows where read)
    if (PBR[p]) begin : g_pbr_chk
      always @(posedge clk) if (chk) begin
        if (r_ben[p] != x_en)
          $fatal(1, "TMEM%0d: port %0d's banks %b are not its lanes' %b at %0t", SID, p, r_ben[p],
                 x_en, $time);
        for (int b = 0; b < LANES; b++)
          if (x_en[b] && r_bad[p][b][BW +: IW] != x_a[b])
            $fatal(1, "TMEM%0d: port %0d's bank %0d row %0h is not its lane's %0h at %0t", SID, p, b,
                   r_bad[p][b][BW +: IW], x_a[b], $time);
      end
    end
`endif
  end

`ifndef SYNTHESIS
  // ---- the dump (simulation): the memory itself, not the writes asked of it. On `dump` the first
  // copy's banks (DP) put their words into img, word i = bank i mod LANES's row i / LANES; a cycle
  // later every other copy is checked against them (a copy that differs stops the simulation)
  // and img goes to <dir>/tmem_<SID>.hex. +tmem_fault=<SID>,<word>: every write to the word
  // lands with bit 0 flipped, in every copy, or with +tmem_fault_port=<port> in that port's
  // (tests: the dump and the check see a fault in the registered write stage, the RAM or one
  // copy)
  function automatic int first_copy();
    for (int p = 0; p < NRP; p++) if (host_of(p) == p) return p;
    return 0;
  endfunction
  localparam int DP = first_copy();
  logic [31:0] img [WORDS];
  logic        dump_q;
  int          fault_w, fault_p;
  initial begin
    string arg;
    int sid, w, p;
    dump_q = 1'b0; fault_w = -1; fault_p = -1;
    if ($value$plusargs("tmem_fault=%s", arg) && $sscanf(arg, "%d,%d", sid, w) == 2 && sid == SID)
    begin
      fault_w = w;
      if ($value$plusargs("tmem_fault_port=%d", p)) fault_p = p;
    end
  end
  always @(posedge clk) dump_q <= dump;
  function automatic logic [31:0] fault(input int p, input int b, input logic [IW-1:0] row);
    return 32'(fault_w >= 0 && fault_w == int'(row) * LANES + b && (fault_p < 0 || fault_p == p));
  endfunction
`endif

  // ---- the copies: one per read port, except the guests (SH_MASK), which read SH_HOST's copy:
  // it serves the first port that requests, the host, then the guests in port order. IMPL 1:
  // one per CPY value, held by its first port; each bank serves the port granted in it.
  // ---- IMPL 3 (EXPERIMENTAL, docs/tmem_nocopy.md): one copy. Each bank is NSB RAMs with their
  // own addresses: a word's sub-bank is its bank row's bits [SBP-BW +: log2 NSB] (SBP = BW: the
  // 8-word groups interleave over the RAMs; higher: regions), XORed with the row's fields above
  // them (SBH 1: the next one, 2: all of them: a hash). A RAM's address is that of the asker
  // first in rank, from the requests (no grant on a RAM's address path, as RQ). An asker that
  // asks a RAM an asker before it took for another word (SHARE: one word read by two is one read)
  // loses it.
  // Without RPL the ranks are PRANK's and a port that would lose is blocked: by a port of
  // another group (blk_x: the slice does not grant the unit) or of its own (blk_s: the slice reads
  // the group's ports over successive cycles, otpu_slice TMEM_IMPL 3).
  // RPL (replay): the units read as if no RAM were shared (the slice's grant ignores the RAMs) and
  // a read that lost is read again the next cycle: the port's banks and rows are registered with
  // the read, and the next cycle they ask again (frz) ranked before every request but TOPP's (the
  // DMA's, which has no hold), and among the replays the oldest first (age, a forward-progress
  // guarantee); their words reach the port's lanes the cycle after (the lanes
  // fresh again, without the bypass: the replay returns what the first read would have, the
  // writes landed since are the ones that were before it). `lost` tells the slice to hold the
  // port's unit for the cycle the replay takes (its lanes' words of the lost read are not the
  // ones asked). blk_x is then a port's request losing (for units that cannot hold, which the
  // slice keeps from reading instead), blk_s 0.
  // A port asks with its request, or with its replay while one is pending (the slice then holds
  // its unit, so its request does not ask): one asker a port, whose rank the replay registers pick.
  // Writes: one per bank and cycle, as IMPL 0 and 1, to the RAM of its sub-bank; the registered
  // write stage and its bypass are IMPL 0's.
  // TDP: true dual-port RAMs, each read port on one of a RAM's two ports (SIDE: 0 port B, 1 port
  // A), ranked among the ports on its side only; a write takes its RAM's port A, so an asker on
  // port A of a RAM being written loses unless it reads the written word (a read takes the
  // bypass, a replay port A's read-first word). On the 7-series a true dual-port RAMB36 is at most
  // 36 bits wide: a RAM of 1K words is one RAMB36, as in the simple dual-port.
  // PMP (with TDP): the RAMs pumped at twice the core clock (1: otpu_pump_bank on clk_f, as P3's
  // copies, docs/tmem_pump.md): both RAM ports read at the core edge and the write lands at the fast
  // edge between, so a write takes no read port (no port-A conflicts) and a read at the core edge
  // sees the landing write in the RAM (no bypass). A replay then must not meet a write of its word
  // landing in its winning cycle either (the ordering of docs/tmem_nocopy.md 5.1 keeps it out). 2:
  // the same at the core clock for simulation (a write port of its own, the bypass kept): the
  // same words and cycles as 1, without the second clock.
  // NSB 1: one RAM a bank (no sub-banks, no data select).
  // GST: IMPL 0's guests (SH_MASK, the slice's QUANT 3 and COLL): a guest reads only while SH_HOST
  // and the guests before it ask nothing (the slice's q3_blk / coll_blk), so it has no asker of its
  // own: SH_HOST's asker asks for the first guest that reads, and the guests take its data (one
  // data select for the three). With RPL the host is TOPP, which never loses.
  // an asker's rank (0 first): without RPL PRANK's; with it TOPP's request, the replays in PRANK
  // order (on a side with three or more replay askers the oldest first: age, below; PRANK's for
  // those begun in one cycle), then the other requests in PRANK order. rlive: the port's reads
  // may be replayed
  function automatic int rnk(input int p, input bit rp);
    if (!RPL) return int'(PRANK[p]);
    if (rp) return 1 + int'(PRANK[p]);
    if (p == TOPP) return 0;
    return 1 + NRP + int'(PRANK[p]);
  endfunction
  function automatic bit rlive(input int p);
    return RPL && RPLM[p] && p != TOPP;
  endfunction
  function automatic int sd_of(input int p);   // the RAM port a read port asks on (0: B)
    return (TDP && SIDE[p]) ? 1 : 0;
  endfunction
  // OTPU_TMEM_ABL_* (EXPERIMENTAL, out-of-context LUT breakdown only, docs/tmem_nocopy.md 6.3: the
  // TMEM is wrong with any of them): _DSEL a port takes two fixed RAMs' words XORed (no data
  // select; every RAM output still used), _CMP no
  // conflict compares (nothing lost), _RPL no replay buffers (an asker is its request), _AMUX a
  // RAM's address is its side's first port's (no priority multiplexer)
  if (IMPL == 3) begin : g_one
    localparam int LSB = $clog2(NSB);
    localparam int LW  = (LSB > 0) ? LSB : 1;  // a sub-bank number's width (NSB 1: always 0)
    localparam int SO  = SBP - BW;             // the sub-bank field's offset in a bank row
    localparam bit BYP = PMP != 1;             // the bypass (a pumped RAM has the landing write)
    localparam int RW  = IW - LSB;             // a RAM's address bits
    localparam int RD  = BD / NSB;             // a RAM's words
    localparam int NS  = TDP ? 2 : 1;          // a RAM's read ports
    function automatic logic [LW-1:0] hash(input logic [IW-1:0] hi);
      logic [LW-1:0] x;
      x = '0;
      if (LSB > 0 && SBH == 1) x = LW'(hi);
      if (LSB > 0 && SBH == 2) for (int f = 0; f < IW; f += LW) x = x ^ LW'(hi >> f);
      return x;
    endfunction
    function automatic logic [RW-1:0] ram_row(input logic [IW-1:0] row);
      logic [IW-1:0] hi, lo;
      hi = row >> (SO + LSB);
      lo = row & ((IW'(1) << SO) - IW'(1));
      return RW'((hi << SO) | lo);
    endfunction
    function automatic logic [LW-1:0] sub_of(input logic [IW-1:0] row);
      if (LSB == 0) return '0;
      return LW'(row >> SO) ^ hash(row >> (SO + LSB));
    endfunction
`ifndef SYNTHESIS
    // the bank row held in RAM k's row r (the dump)
    function automatic int row_of(input int k, input int r);
      logic [IW-1:0] hi, lo, f;
      hi = IW'(r >> SO);
      lo = IW'(r & ((1 << SO) - 1));
      f = (LSB == 0) ? '0 : IW'(LW'(k) ^ hash(hi));
      return int'((hi << (SO + LSB)) | (f << SO) | lo);
    endfunction
`endif
    // the replays: a lost read's banks, rows and lanes, registered with it
    logic [NRP-1:0]                    rpv;
    // ... and rpv's copy per bank (rpvb: kept, each loaded with lost as rpv is): bank b's asker
    // multiplexers, its replay registers' loads, its data select and its lane's replay take theirs,
    // so no one register's replicas reach every bank's (r150c, the m8 card at a real 150 MHz:
    // rpv's replica, fanout 42, into the pumped RAMs' read pins +0.041 ns, into lost_q +0.028)
    (* keep = "true" *) logic [NRP-1:0][LANES-1:0] rpvb;
    logic [NRP-1:0][LANES-1:0]         rpe;
    logic [NRP-1:0][LANES-1:0][IW-1:0] rpa;
    logic [NRP-1:0][LANES-1:0]         rpl;
    logic [NRP-1:0]                    rp;      // the port's asker is its replay
    // each port's requested banks and rows (GST: the host's are the first reading guest's while
    // it asks nothing itself; a guest asks nothing)
    logic [NRP-1:0][LANES-1:0]         qe;
    logic [NRP-1:0][LANES-1:0][IW-1:0] qa;
    always_comb begin
      logic take;
      qe = pb_en; qa = pb_a;
      take = |r_req[SH_HOST];
      if (GST)
        for (int g = 0; g < NRP; g++)
          if (SH_MASK[g]) begin
            if (!take && |r_req[g]) begin
              qe[SH_HOST] = pb_en[g]; qa[SH_HOST] = pb_a[g]; take = 1'b1;
            end
            qe[g] = '0;
          end
    end
    // each asker's banks, bank rows, sub-banks and RAM rows
    logic [NRP-1:0][LANES-1:0]          ae;
    logic [NRP-1:0][LANES-1:0][IW-1:0]  aa;
    logic [NRP-1:0][LANES-1:0][LW-1:0]  asb;
    logic [NRP-1:0][LANES-1:0][RW-1:0]  arw;
    for (genvar i = 0; i < NRP; i++) begin : g_as
`ifdef OTPU_TMEM_ABL_RPL
      assign rp[i] = 1'b0;
`else
      assign rp[i] = rlive(i) && rpv[i];
`endif
      for (genvar b = 0; b < LANES; b++) begin : g_b
`ifdef OTPU_TMEM_ABL_RPL
        wire rpb = 1'b0;
`else
        wire rpb = rlive(i) && rpvb[i][b];         // rp[i], bank b's copy
`endif
        assign ae[i][b] = rpb ? rpe[i][b] : qe[i][b];
        assign aa[i][b] = rpb ? rpa[i][b] : qa[i][b];
        assign asb[i][b] = sub_of(aa[i][b]);
        assign arw[i][b] = ram_row(aa[i][b]);
      end
    end
    // The replays' age (RPL, a forward-progress guarantee): of two replays the older ranks first
    // (begun in one cycle, in PRANK order). A replay then loses only to TOPP's ask and to replays
    // older than it, and the oldest wins as soon as TOPP leaves its RAMs. A fixed order let two
    // ports that took turns replaying hold off a third: each lost a read to the third's pending
    // replay (which took that RAM, losing another) and its replay then beat the third's. With two
    // replay askers on a side that needs TOPP every other cycle, so the age is kept only on a side
    // of three or more (the pick's port B: the MXU, VPU B, QUANT 2; GST's guests have no asker).
    function automatic bit asker_rl(input int p);   // a replayable port with an asker of its own
      return rlive(p) && !(GST && SH_MASK[p]);
    endfunction
    function automatic bit aged(input int i, input int o);
      int n;
      n = 0;
      for (int p = 0; p < NRP; p++) if (asker_rl(p) && sd_of(p) == sd_of(i)) n++;
      return i != o && asker_rl(i) && asker_rl(o) && sd_of(i) == sd_of(o) && n >= 3;
    endfunction
    // age[i][o] (o < i): o's replay began before i's (meaningful while both replay). A replay
    // begins the cycle after its port's read loses (lost, rpv low): o's beginning makes it the
    // younger, unless i's begins with it
    logic [NRP-1:0][NRP-1:0] age;
    always_ff @(posedge clk)
      for (int i = 0; i < NRP; i++)
        for (int o = 0; o < i; o++)
          if (aged(i, o)) begin
            if (rst) age[i][o] <= 1'b0;
            else if (lost[o] && !rpv[o])
              age[i][o] <= (lost[i] && !rpv[i]) ? rnk(o, 1'b1) < rnk(i, 1'b1) : 1'b0;
            else if (lost[i] && !rpv[i]) age[i][o] <= 1'b1;
          end
    // bf[i][o]: asker o ranks before asker i this cycle (on i's side; the replays' registers pick
    // the case of four: three constant ones, and with both replaying their age where it is kept)
    logic [NRP-1:0][NRP-1:0] bf;
    for (genvar i = 0; i < NRP; i++) begin : g_bf
      for (genvar o = 0; o < NRP; o++) begin : g_o
        if (o == i || sd_of(o) != sd_of(i)) begin : g_n
          assign bf[i][o] = 1'b0;
        end else begin : g_y
          localparam bit B00 = rnk(o, 1'b0) < rnk(i, 1'b0);
          localparam bit B01 = rnk(o, 1'b0) < rnk(i, 1'b1);
          localparam bit B10 = rnk(o, 1'b1) < rnk(i, 1'b0);
          localparam bit B11 = rnk(o, 1'b1) < rnk(i, 1'b1);
          wire ol = !aged(i, o) ? B11 : (o < i) ? age[i][o] : !age[o][i];
          assign bf[i][o] = rp[o] ? (rp[i] ? ol : B10) : (rp[i] ? B01 : B00);
        end
      end
    end
    // wk[i][b]: TDP, asker i asks port A of a RAM the write in bank b takes
    logic [NRP-1:0][LANES-1:0] wk;
    for (genvar i = 0; i < NRP; i++) begin : g_wk
      for (genvar b = 0; b < LANES; b++) begin : g_b
        if (sd_of(i) == 1 && PMP == 0) begin : g_a
          assign wk[i][b] = pw_v[b][0] && sub_of(pw_a[b][0]) == asb[i][b];
        end else begin : g_b0
          assign wk[i][b] = 1'b0;
        end
      end
    end
    // the RAMs' addresses: per RAM and side, the asker first in rank (an AND-OR multiplexer of
    // the askers' rows)
    logic [LANES-1:0][NSB-1:0][NS-1:0]         s_en;
    logic [LANES-1:0][NSB-1:0][NS-1:0][RW-1:0] s_a;
    for (genvar b = 0; b < LANES; b++) begin : g_ra
      for (genvar k = 0; k < NSB; k++) begin : g_k
        logic [NRP-1:0] rq, win;
        for (genvar i = 0; i < NRP; i++) begin : g_i
          assign rq[i] = ae[i][b] && asb[i][b] == LW'(k);
          assign win[i] = rq[i] && !(|(rq & bf[i]));
        end
        always_comb
          for (int s = 0; s < NS; s++) begin
            bit got;
            got = 1'b0;
            s_en[b][k][s] = 1'b0;
            s_a[b][k][s] = '0;
            for (int i = 0; i < NRP; i++)
              if (sd_of(i) == s) begin
                s_en[b][k][s] = s_en[b][k][s] | rq[i];
`ifdef OTPU_TMEM_ABL_AMUX
                if (!got) s_a[b][k][s] = arw[i][b];
                got = 1'b1;
`else
                if (win[i]) s_a[b][k][s] = s_a[b][k][s] | arw[i][b];
`endif
              end
          end
      end
    end
    // cp[i][o] (o < i, one side): the two askers ask one RAM for different words; hm[i][o]: o,
    // ranked before i, does so (on port A of a RAM being written the write decides: wm)
    logic [NRP-1:0][NRP-1:0] cp, hm;
    logic [NRP-1:0]          wm;
    for (genvar i = 0; i < NRP; i++) begin : g_cf
      for (genvar o = 0; o < NRP; o++) begin : g_o
        if (o < i && sd_of(o) == sd_of(i)) begin : g_c
          logic [LANES-1:0] hb;
          for (genvar b = 0; b < LANES; b++) begin : g_b
            assign hb[b] = ae[i][b] && ae[o][b] && asb[o][b] == asb[i][b] && !wk[i][b] &&
                           (!SHARE || aa[o][b] != aa[i][b]);
          end
`ifdef OTPU_TMEM_ABL_CMP
          assign cp[i][o] = 1'b0;
`else
          assign cp[i][o] = |hb;
`endif
        end else begin : g_n
          assign cp[i][o] = 1'b0;
        end
      end
      logic [LANES-1:0] wb;
      for (genvar b = 0; b < LANES; b++) begin : g_w
        assign wb[b] = ae[i][b] && wk[i][b] && pw_a[b][0] != aa[i][b];
      end
`ifdef OTPU_TMEM_ABL_CMP
      assign wm[i] = 1'b0;
`else
      assign wm[i] = |wb;
`endif
    end
    always_comb
      for (int i = 0; i < NRP; i++)
        for (int o = 0; o < NRP; o++)
          hm[i][o] = (o < i ? cp[i][o] : cp[o][i]) && bf[i][o];
    always_comb
      for (int p = 0; p < NRP; p++) begin
        blk_x[p] = wm[p]; blk_s[p] = 1'b0;
        for (int o = 0; o < NRP; o++)
          if (hm[p][o]) begin
            if (!RPL && GRP[o] == GRP[p]) blk_s[p] = 1'b1;
            else blk_x[p] = 1'b1;
          end
      end
    // a read lost: the port read (its unit was granted, r_en) or replayed, and lost a RAM
    always_comb
      for (int p = 0; p < NRP; p++)
        lost[p] = rlive(p) && (rpv[p] || (|r_en[p])) && ((|hm[p]) || wm[p]);
    always_ff @(posedge clk)
      for (int p = 0; p < NRP; p++) begin
        rpv[p] <= rst ? 1'b0 : lost[p];
        for (int b = 0; b < LANES; b++) begin
          rpvb[p][b] <= rst ? 1'b0 : lost[p];
          if (!rpvb[p][b]) begin rpe[p][b] <= qe[p][b]; rpa[p][b] <= qa[p][b]; rpl[p][b] <= r_en[p][b]; end
        end
      end
`ifndef SYNTHESIS
    always @(posedge clk)
      for (int p = 0; p < NRP; p++)
        if (chk && rpvb[p] != {LANES{rpv[p]}}) $fatal(1, "TMEM%0d: port %0d's rpv copies %b are not rpv %b", SID, p,
                                               rpvb[p], rpv[p]);
    // the replays' premise: no write lands on a word awaiting its replay before the replay wins
    // (it would read the new word). A write landing in the cycle the replay wins is fine (the
    // RAM reads first); one landing while it loses again is not.
    always @(posedge clk)
      for (int p = 0; p < NRP; p++)
        if (chk && RPL && rpv[p] && (lost[p] || PMP == 1))
          for (int b = 0; b < LANES; b++)
            for (int w = 0; w < WPB; w++)
              if (rpe[p][b] && pw_v[b][w] && pw_a[b][w] == rpa[p][b])
                $fatal(1, "TMEM%0d: port %0d's word %0d awaiting a replay was written at %0t", SID,
                       p, int'(rpa[p][b]) * LANES + b, $time);
    // forward progress: a replay that loses RPLW cycles in a row stops the simulation (a livelock:
    // what holds a replay off, TOPP's asks and older replays, lasts that long only when it waits on
    // the replaying unit itself)
    localparam int RPLW = 1 << 22;
    int rpn [NRP];
    initial for (int p = 0; p < NRP; p++) rpn[p] = 0;
    always @(posedge clk)
      for (int p = 0; p < NRP; p++) begin
        rpn[p] <= (RPL && rpv[p] && lost[p]) ? rpn[p] + 1 : 0;
        if (rpn[p] >= RPLW)
          $fatal(1, "TMEM%0d: port %0d's replay lost %0d cycles in a row at %0t", SID, p, rpn[p],
                 $time);
      end
`endif
    assign frz = RPL ? rpv : '0;
    always_comb
      for (int p = 0; p < NRP; p++)
        for (int l = 0; l < LANES; l++) rpf[p][l] = RPL && rpvb[p][l] && rpl[p][l];
    logic [LANES-1:0][NSB-1:0][NS-1:0][31:0] s_q;
    for (genvar b = 0; b < LANES; b++) begin : g_bank
      for (genvar k = 0; k < NSB; k++) begin : g_sub
        logic [31:0] wd, wd1;
`ifdef SYNTHESIS
        assign wd = pw_d[b][0];
        assign wd1 = pw_d[b][WPB-1];
`else
        assign wd = pw_d[b][0] ^ fault(0, b, pw_a[b][0]);
        assign wd1 = pw_d[b][WPB-1] ^ fault(0, b, pw_a[b][WPB-1]);
`endif
        wire wh = pw_v[b][0] && sub_of(pw_a[b][0]) == LW'(k);
        // WPB 2 (pumped): the bank's second write, on the RAM's port B at the odd edge
        wire wh1 = WPB == 2 && pw_v[b][WPB-1] && sub_of(pw_a[b][WPB-1]) == LW'(k);
`ifndef SYNTHESIS
        always @(posedge clk)
          if (wh && wh1 && pw_a[b][0] == pw_a[b][WPB-1])
            $fatal(1, "TMEM%0d: two writes of word %0d in one cycle at %0t", SID,
                   int'(pw_a[b][0]) * LANES + b, $time);
`endif
        // (inferred here: 64 otpu_ram_1r1w cells of this size in the slice crash Verilator 5.046's
        // front end, an internal fault; the cell is this RAM)
        if (PMP == 1) begin : g_pmp
          // pumped at 2x: port A reads side 1, port B side 0 at the core edge, the write lands at
          // the fast edge between (P3's bank). The write's inputs are this RAM's own copy of the
          // registered write stage (pk_*, release r2): its sub-bank's decode and the RAM row are
          // taken before the register, so the bank's write pins (on clk_f, a half core cycle
          // from the register) come straight from flip-flops kept next to it, not through the
          // bank-level pw's decode and fanout (r2m4: pw_* -> the pumped banks, clk -> clk2x).
          // The RAM is PSN slices of PSW bits, each one block RAM at RD rows (RAMB36: 32K bits),
          // each with its own copy of the write stage and its own phase (u_pb): a register drives
          // one RAM's write pins, not the PSN RAMs Vivado tiles 32 bits into (r22's m8: one pk_a
          // through the ad_a mux to 8 RAMs spread over 22 RAM rows, clk -> clk2x -0.202 ns).
          // STYLE "ultra" (the WG targets'): an UltraRAM is 72 bits wide at any depth (4K rows), so the
          // RAM is one slice of the whole word (narrower slices would each take whole UltraRAMs)
          localparam int PSW = (STYLE == "ultra") ? 32 :
                               (RD >= 32768) ? 1 : (32768 / RD >= 32) ? 32 : 32768 / RD;
          localparam int PSN = 32 / PSW;
          logic [PSN-1:0]           pk_v, pk_v1;
          logic [PSN-1:0][RW-1:0]   pk_a, pk_a1;
          logic [31:0]              pk_d, pkd, pk_d1, pkd1;
          for (genvar z = 0; z < PSN; z++) begin : g_sl
            (* keep = "true" *) logic           sv;
            (* keep = "true" *) logic [RW-1:0]  sa;
            (* keep = "true" *) logic [PSW-1:0] sd;
            always_ff @(posedge clk) begin
              sv <= rst ? 1'b0 : bw_v[b][0] && sub_of(bw_a[b][0]) == LW'(k);
              sa <= ram_row(bw_a[b][0]);
              sd <= bw_d[b][0][z*PSW +: PSW];
            end
            assign pk_v[z] = sv;
            assign pk_a[z] = sa;
            assign pk_d[z*PSW +: PSW] = sd;
            // WPB 2: the second write's own copy of the stage (port B's pins)
            logic           sv1;
            logic [RW-1:0]  sa1;
            logic [PSW-1:0] sd1;
            if (WPB == 2) begin : g_w2
              (* keep = "true" *) logic           v1;
              (* keep = "true" *) logic [RW-1:0]  a1;
              (* keep = "true" *) logic [PSW-1:0] d1;
              always_ff @(posedge clk) begin
                v1 <= rst ? 1'b0 : bw_v[b][WPB-1] && sub_of(bw_a[b][WPB-1]) == LW'(k);
                a1 <= ram_row(bw_a[b][WPB-1]);
                d1 <= bw_d[b][WPB-1][z*PSW +: PSW];
              end
              assign sv1 = v1, sa1 = a1, sd1 = d1;
            end else begin : g_w1
              assign sv1 = 1'b0, sa1 = '0, sd1 = '0;
            end
            assign pk_v1[z] = sv1;
            assign pk_a1[z] = sa1;
            assign pk_d1[z*PSW +: PSW] = sd1;
            otpu_pump_bank #(.N(RD), .W(PSW), .STYLE(STYLE), .FW(1'b1), .W2(WPB == 2)) u_pb (
              .clk, .clk_f, .rst, .wv(sv), .wa(sa), .wd(pkd[z*PSW +: PSW]),
              .wv2(sv1), .wa2(sa1), .wd2(pkd1[z*PSW +: PSW]),
              .rena(s_en[b][k][1]), .ra(s_a[b][k][1]), .renb(s_en[b][k][0]), .rb(s_a[b][k][0]),
              .qa(s_q[b][k][1][z*PSW +: PSW]), .qb(s_q[b][k][0][z*PSW +: PSW]));
          end
`ifdef SYNTHESIS
          assign pkd = pk_d;
          assign pkd1 = pk_d1;
`else
          assign pkd = pk_d ^ fault(0, b, pw_a[b][0]);
          assign pkd1 = pk_d1 ^ fault(0, b, pw_a[b][WPB-1]);
          // every slice's copy is the bank's write stage at this RAM
          always @(posedge clk)
            for (int z = 0; z < PSN; z++)
              if (chk && (pk_v[z] != wh ||
                          (wh && (pk_a[z] != ram_row(pw_a[b][0]) || pk_d != pw_d[b][0])) ||
                          pk_v1[z] != wh1 ||
                          (wh1 && (pk_a1[z] != ram_row(pw_a[b][WPB-1]) || pk_d1 != pw_d[b][WPB-1]))))
                $fatal(1, "TMEM%0d: bank %0d RAM %0d slice %0d's write copy differs from the write stage",
                       SID, b, k, z);
`endif
`ifndef SYNTHESIS
          // the dump: a shadow of the RAM's words (the same writes), not a reference into u_pb
          logic [31:0] shd [RD];
          initial for (int r = 0; r < RD; r++) shd[r] = '0;
          always @(posedge clk) begin
            if (dump)
              for (int r = 0; r < RD; r++) img[row_of(k, r) * LANES + b] = shd[r];
            if (pk_v[0]) shd[pk_a[0]] <= pkd;
            if (pk_v1[0]) shd[pk_a1[0]] <= pkd1;
          end
`endif
        end else begin : g_ram
        (* ram_style = "block" *) logic [31:0] mem [RD];
        if (PMP == 2) begin : g_pm2
          // the model of PMP 1 at the core clock: the write on a port of its own, both read ports
          always_ff @(posedge clk) begin
            if (wh) mem[ram_row(pw_a[b][0])] <= wd;
            if (wh1) mem[ram_row(pw_a[b][WPB-1])] <= wd1;
            if (s_en[b][k][1]) s_q[b][k][1] <= mem[s_a[b][k][1]];
          end
          always_ff @(posedge clk)
            if (s_en[b][k][0]) s_q[b][k][0] <= mem[s_a[b][k][0]];
        end else if (TDP) begin : g_tdp
          // port A: the write, else its side's read; read-first (a replay of the written word
          // takes the word before the write)
          wire [RW-1:0] ra = wh ? ram_row(pw_a[b][0]) : s_a[b][k][1];
`ifdef SYNTHESIS
          // synthesis: the technology RAM (rtl/tech: inferred alike on an FPGA, the 1RW1R
          // macros on an ASIC, which leave mem empty); simulation keeps it inferred here (above)
          otpu_ram_1rw1r #(.W(32), .N(RD), .STYLE("block")) u_ram (
            .clk, .ena(wh || s_en[b][k][1]), .wea(wh), .aa(ra), .wda(wd), .rda(s_q[b][k][1]),
            .enb(s_en[b][k][0]), .ab(s_a[b][k][0]), .rdb(s_q[b][k][0]));
`else
          always_ff @(posedge clk)
            if (wh || s_en[b][k][1]) begin
              if (wh) mem[ra] <= wd;
              s_q[b][k][1] <= mem[ra];
            end
          always_ff @(posedge clk)
            if (s_en[b][k][0]) s_q[b][k][0] <= mem[s_a[b][k][0]];
`endif
        end else begin : g_sdp
          always_ff @(posedge clk) begin
            if (wh) mem[ram_row(pw_a[b][0])] <= wd;
            if (s_en[b][k][0]) s_q[b][k][0] <= mem[s_a[b][k][0]];
          end
        end
`ifndef SYNTHESIS
        initial for (int r = 0; r < RD; r++) mem[r] = '0;
        always @(posedge clk)
          if (dump)
            for (int r = 0; r < RD; r++) img[row_of(k, r) * LANES + b] = mem[r];
`endif
        end
      end
    end
    // each port's data: per bank, the RAM it asked on its side (registered with the read), or the
    // write still in the registered stage that its word hit (a replay: the RAM's word, see above)
    for (genvar p = 0; p < NRP; p++) begin : g_pq
     if (GST && SH_MASK[p]) begin : g_gu
      assign pq[p] = pq[SH_HOST];
      assign pr[p] = '0;
      assign pss[p] = '0;
      assign pdir[p] = 1'b0;
     end else begin : g_ho
      localparam int SD = sd_of(p);
      logic [LANES-1:0][LW-1:0]  ss;
      logic [LANES-1:0]          bh, bh1;
      for (genvar b = 0; b < LANES; b++) begin : g_pr
        for (genvar k = 0; k < NSB; k++) begin : g_k
          assign pr[p][b * NSB + k] = s_q[b][k][SD];
        end
        assign pss[p][b] = PSB'(ss[b]);
      end
`ifdef OTPU_TMEM_ABL_DSEL
      assign pdir[p] = 1'b0;
`else
      assign pdir[p] = !BYP;
`endif
      // (the write hits, bh / bh1, reset: the read's word select)
      always_ff @(posedge clk)
        for (int b = 0; b < LANES; b++) begin
`ifdef OTPU_TMEM_ABL_RPL
          if (1'b0) begin
`else
          if (RPL && rpvb[p][b] && rpe[p][b]) begin
`endif
            ss[b] <= sub_of(rpa[p][b]);
            bh[b] <= 1'b0;
            bh1[b] <= 1'b0;
          end else if (qe[p][b]) begin
            ss[b] <= sub_of(qa[p][b]);
            bh[b] <= BYP && pw_v[b][0] && pw_a[b][0] == qa[p][b];
            bh1[b] <= BYP && WPB == 2 && pw_v[b][WPB-1] && pw_a[b][WPB-1] == qa[p][b];
          end
          if (rst) begin
            bh[b] <= 1'b0;
            bh1[b] <= 1'b0;
          end
        end
      always_comb
`ifdef OTPU_TMEM_ABL_DSEL
        for (int b = 0; b < LANES; b++)
          pq[p][b] = bh[b] ? pd[b][0] : s_q[b][p % NSB][SD] ^ s_q[b][(p + NSB / 2) % NSB][SD];
`else
        for (int b = 0; b < LANES; b++)
          pq[p][b] = bh[b] ? pd[b][0] : bh1[b] ? pd[b][WPB-1] : s_q[b][ss[b]][SD];
`endif
     end
    end
  end else begin : g_nosb
    assign blk_x = '0;
    assign blk_s = '0;
    assign lost = '0;
    assign frz = '0;
    assign rpf = '0;
    for (genvar p = 0; p < NRP; p++) begin : g_npr
      assign pr[p] = '0;
    end
    assign pss = '0;
    assign pdir = '0;
  end
  for (genvar p = 0; p < NRP; p++) begin : g_port
   if (IMPL != 3 && host_of(p) == p) begin : g_copy
    logic [LANES-1:0]         b_en;
    logic [LANES-1:0][IW-1:0] b_a;
    if (IMPL == 1 && RQ) begin : g_bq
      // per bank, the asking port on this copy first in priority (ranks are distinct per copy)
      always_comb begin
        b_en = '0; b_a = '0;
        for (int b = 0; b < LANES; b++)
          for (int r = NRP - 1; r >= 0; r--)
            for (int g = 0; g < NRP; g++)
              if (CPY[g] == CPY[p] && int'(PRANK[g]) == r && pb_en[g][b]) begin
                b_en[b] = 1'b1;
                b_a[b] = pb_a[g][b];
              end
      end
    end else if (IMPL == 1) begin : g_bk
      always_comb begin
        b_en = '0; b_a = '0;
        for (int g = 0; g < NRP; g++)
          if (CPY[g] == CPY[p])
            for (int b = 0; b < LANES; b++)
              if (pb_en[g][b]) begin
                b_en[b] = 1'b1;
                b_a[b] = b_a[b] | pb_a[g][b];
              end
      end
    end else if (p == SH_HOST && SH_MASK != '0) begin : g_sh
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
      logic [31:0] mem [BD];               // IMPL 0's (IMPL 1: unused, the RAM cell holds them)
      if (IMPL == 1) begin : g_ram
        logic [31:0] wd;
`ifdef SYNTHESIS
        assign wd = pw_d[b][0];
`else
        assign wd = pw_d[b][0] ^ fault(p, b, pw_a[b][0]);
`endif
        otpu_ram_1r1w #(.W(32), .N(BD), .STYLE("block")) u_ram (
          .clk, .we(pw_v[b][0]), .wa(pw_a[b][0]), .wd, .re(b_en[b]), .ra(b_a[b]), .rd(q[b]));
      end else begin : g_inf
        always_ff @(posedge clk) begin
          if (b_en[b]) q[b] <= mem[b_a[b]];
          for (int w = 0; w < WPB; w++)
`ifdef SYNTHESIS
            if (pw_v[b][w]) mem[pw_a[b][w]] <= pw_d[b][w];
`else
            if (pw_v[b][w]) mem[pw_a[b][w]] <= pw_d[b][w] ^ fault(p, b, pw_a[b][w]);
`endif
        end
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
      // the dump: DP's words, then the others against them
      if (IMPL == 1) begin : g_dump1
        always @(posedge clk) begin
          if (dump && p == DP)
            for (int r = 0; r < BD; r++) img[r * LANES + b] = g_ram.u_ram.g_bram.mem[r];
          if (dump_q && p != DP)
            for (int r = 0; r < BD; r++)
              if (g_ram.u_ram.g_bram.mem[r] != img[r * LANES + b])
                $fatal(1, "TMEM%0d: copy %0d's word %0d is %h, copy %0d's %h", SID, p,
                       r * LANES + b, g_ram.u_ram.g_bram.mem[r], DP, img[r * LANES + b]);
        end
      end else begin : g_dump0
        always @(posedge clk) begin
          if (dump && p == DP)
            for (int r = 0; r < BD; r++) img[r * LANES + b] = mem[r];
          if (dump_q && p != DP)
            for (int r = 0; r < BD; r++)
              if (mem[r] != img[r * LANES + b])
                $fatal(1, "TMEM%0d: copy %0d's word %0d is %h, copy %0d's %h", SID, p,
                       r * LANES + b, mem[r], DP, img[r * LANES + b]);
        end
      end
`endif
    end
    assign pq[p] = qb;
   end else if (IMPL != 3) begin : g_guest
    assign pq[p] = pq[host_of(p)];
   end
    wire  [BW-1:0]            rot = r_addr[p][0][BW-1:0];
    wire  [LANES-1:0][31:0]   qb = pq[p];
    // lane data: the bank it read (the crossbar: each lane's bank; a rotator: the run's
    // rotation), fresh the cycle after the read, then held. A rotator keeps the rotation once
    // per lane (rot_q[l], all loaded with the run's): lane l takes bank (l + rot_q[l]) mod LANES,
    // so each copy drives one lane's select and hold, not every lane's (133.33 MHz, 110ec6d /
    // 812bb01: g_port[6].rot_q[2] -> 1 LUT -> held, fanout 513, 6.8 ns of route, +0.152 ns)
    logic [LANES-1:0][BW-1:0] sel;
    logic [LANES-1:0][BW-1:0] rq;         // a rotator's rotation, per lane
    logic [LANES-1:0]         fresh;
    logic [LANES-1:0][31:0]   held, lq;
    if (GEN_R[p]) begin : g_xb
      assign rq = '0;
    end else begin : g_rot
      (* keep *) logic [LANES-1:0][BW-1:0] rot_q;
      always_ff @(posedge clk)
        for (int l = 0; l < LANES; l++) if (|r_en[p]) rot_q[l] <= rot;
      assign rq = rot_q;
`ifndef SYNTHESIS
      // the lanes' copies are one rotation (the run's), from the first read on
      bit ld;
      initial ld = 1'b0;
      always @(posedge clk) begin
        if (chk && |r_en[p]) ld <= 1'b1;
        for (int l = 1; l < LANES; l++)
          if (ld && rot_q[l] != rot_q[0])
            $fatal(1, "otpu_tmem: port %0d lane %0d's rotation %0d is not lane 0's %0d", p, l,
                   rot_q[l], rot_q[0]);
      end
`endif
    end
    always_ff @(posedge clk)
      for (int l = 0; l < LANES; l++) begin
        fresh[l] <= r_en[p][l] | rpf[p][l];   // (rpf: IMPL 3 RPL's replays)
        if (GEN_R[p] && r_en[p][l]) sel[l] <= r_addr[p][l][BW-1:0];
        if (fresh[l]) held[l] <= lq[l];
      end
    logic [LANES-1:0][BW-1:0] bk;          // the lane's bank
    always_comb
      for (int l = 0; l < LANES; l++) begin
        bk[l] = GEN_R[p] ? sel[l] : BW'((l + int'(rq[l])) % LANES);
        // pdir: the word by {bank, the bank's RAM}, both from flip-flops (pq's select in the index)
        lq[l] = pdir[p] ? pr[p][int'(bk[l]) * NSB + int'(pss[p][bk[l]])] : qb[bk[l]];
        r_data[p][l] = fresh[l] ? lq[l] : held[l];
      end
`ifndef SYNTHESIS
    // the select by {bank, RAM} is the bank's word (pq), lane by lane
    always @(posedge clk)
      for (int l = 0; l < LANES; l++)
        if (chk && pdir[p] && fresh[l] && lq[l] !== qb[bk[l]])
          $fatal(1, "TMEM%0d: port %0d lane %0d's word %h is not bank %0d's %h", SID, p, l, lq[l],
                 bk[l], qb[bk[l]]);
`endif
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
  always @(posedge clk) if (chk) begin
    for (int p = 0; p < NRP; p++)
      if (IMPL == 0 && SH_MASK[p] && |r_en[p]) begin
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
    for (int p = 0; p < NWP; p++)      // (the rotators' writes: WPB 1, and IMPL 3's WPB 2, g_w1)
      if ((WPB == 1 || IMPL == 3) && !GEN_W[p] && !contiguous(w_req[p], w_addr[p]))
        $fatal(1, "TMEM%0d write port %0d: lanes not contiguous at %0t", SID, p, $time);
    for (int p = 0; p < NRP; p++)
      for (int l = 0; l < LANES; l++)
        if (r_en[p][l] && r_addr[p][l] >= WORDS) $fatal(1, "TMEM%0d read beyond %0d words", SID, WORDS);
    for (int p = 0; p < NWP; p++)
      for (int l = 0; l < LANES; l++)
        if (w_en[p][l] && w_addr[p][l] >= WORDS) $fatal(1, "TMEM%0d write beyond %0d words", SID, WORDS);
  end
  string dir;
  initial void'($value$plusargs("dir=%s", dir));
  always @(posedge clk) if (dump_q) begin
    int fd;
    fd = $fopen($sformatf("%s/tmem_%0d.hex", dir, SID), "w");
    for (int i = 0; i < WORDS; i++) $fwrite(fd, "%08x\n", img[i]);
    $fclose(fd);
  end
`endif
endmodule

// One bank of a pumped copy: a true dual-port RAM on clk_f (= 2 x clk, aligned). Even fast edges
// (the core edges): port A reads ra (rena), port B reads rb (renb). Odd fast edges (mid core
// cycle): port A writes wd at wa (wv), its output held (NO_CHANGE); W2, port B writes wd2 at wa2
// (wv2, another word). qa / qb change only after even edges and hold through the next core cycle. Its hierarchy is kept: the timing contract names its
// read inputs and its outputs (boards/ypcb-00338/constraints/otpu_clk2x.tcl, both boards' and the
// out-of-context runs' with the one copy; rtl/exp/tmem_pump/ooc.tcl's for P3). Here since the one
// copy (otpu_tmem IMPL 3 PMP 1) builds it; P3 (rtl/exp/tmem_pump) too.
(* keep_hierarchy = "yes" *)
module otpu_pump_bank #(
  parameter int    N     = 8192,
  parameter int    W     = 32,       // the word's bits (otpu_tmem: a slice of the 32, one RAM)
  parameter string STYLE = "block",
  parameter bit    FW    = 1'b1,     // the phase from a fast-domain register (below)
  parameter bit    W2    = 1'b0      // a second write (wv2), port B's at the odd edges
) (
  input  logic                 clk,
  input  logic                 clk_f,
  input  logic                 rst,      // synchronous to clk: the phase (below)
  input  logic                 wv,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 wv2,      // W2 (otpu_tmem WPB 2): another word's write
  input  logic [$clog2(N)-1:0] wa2,
  input  logic [W-1:0]         wd2,
  input  logic                 rena,
  input  logic [$clog2(N)-1:0] ra,
  input  logic                 renb,
  input  logic [$clog2(N)-1:0] rb,
  output logic [W-1:0]         qa,
  output logic [W-1:0]         qb
);
  // the phase: tog flips at every core edge; tq is its copy a fast edge later. They differ during
  // a core cycle's first half (the next fast edge is odd) and agree in its second (even next).
  // FW: ev is a fast-domain register loaded with that at every fast edge (the edge after next's
  // parity is the opposite of the next's), so no core-domain net reaches the RAM's enables; it
  // aligns itself at the first fast edge, as tq does. The reset (tog is its own next value: from
  // an unknown power-up it would stay unknown) holds the state just after an even edge where tog
  // fell (tog low, tq high, evr low), so the phase is right from the first fast edge after it.
  (* keep *) logic tog;
  (* keep *) logic tq;
  logic evr;
  always_ff @(posedge clk) tog <= rst ? 1'b0 : !tog;
  always_ff @(posedge clk_f) begin
    tq <= rst ? 1'b1 : tog;
    evr <= rst ? 1'b0 : tq != tog;
  end
  wire ev = FW ? evr : (tq == tog);
  wire                 en_a = ev ? rena : wv;
  wire                 we_a = !ev && wv;
  wire [$clog2(N)-1:0] ad_a = ev ? ra : wa;
  wire                 en_b = ev ? renb : W2 && wv2;
  wire                 we_b = W2 && !ev && wv2;
  wire [$clog2(N)-1:0] ad_b = (W2 && !ev) ? wa2 : rb;
  if (STYLE == "ultra") begin : g_u
    (* ram_style = "ultra" *) logic [W-1:0] mem [N];
    always_ff @(posedge clk_f)
      if (en_a) begin
        if (we_a) mem[ad_a] <= wd;
        else qa <= mem[ad_a];
      end
    always_ff @(posedge clk_f)
      if (en_b) begin
        if (we_b) mem[ad_b] <= wd2;
        else qb <= mem[ad_b];
      end
`ifndef SYNTHESIS
    initial for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else begin : g_b
    (* ram_style = "block" *) logic [W-1:0] mem [N];
    always_ff @(posedge clk_f)
      if (en_a) begin
        if (we_a) mem[ad_a] <= wd;
        else qa <= mem[ad_a];
      end
    always_ff @(posedge clk_f)
      if (en_b) begin
        if (we_b) mem[ad_b] <= wd2;
        else qb <= mem[ad_b];
      end
`ifndef SYNTHESIS
    initial for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end
endmodule
