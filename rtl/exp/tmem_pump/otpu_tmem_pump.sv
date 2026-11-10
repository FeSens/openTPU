// EXPERIMENTAL (docs/tmem_pump.md; in no file list): otpu_tmem with its block RAM copies pumped
// at twice the core clock. clk_f = 2 x clk (the same MMCM, both at phase 0: every core edge is a
// fast edge). Each pumped copy is LANES banks of one true dual-port RAM on clk_f; per core cycle
// and bank it serves:
//   - at the core edge (an even fast edge): two reads, port A and port B, from the read ports'
//     requests of the cycle that ends there -- the same edge otpu_tmem's copies read at;
//   - at the fast edge in the middle of the core cycle (odd): port A writes the registered write
//     stage's word (pw_*), which otpu_tmem lands at the next core edge.
// So one pumped copy holds two read slots (A, B) and the write: NS slots need NPC = NS / 2 copies.
// A read's data is a RAM output that holds for the whole next core cycle (port A's write is
// NO_CHANGE, port B only reads), as otpu_tmem's do: the units see the same cycle. A third read
// per copy cannot keep that: it would be issued half a cycle early (before its request settles)
// or its data would land half a cycle late; see docs/tmem_pump.md.
//
// K = 3 or 4 (one copy pumped at 3x / 4x, IMPL 1 with NCP = 2K - 1 slots): see otpu_pumpk_bank.
// More edges do not give more reads at otpu_tmem's latency: a read's word must hold through the
// whole next core cycle, and a port's output holds one word, so each port gives one such read a
// core cycle at any K; a read at any other fast edge either takes its request before it has
// settled or delivers its word late. So the K copy registers the requests at the core edge and
// spreads the reads over the next core cycle: 2K - 1 read slots and the write, each read's word
// presented a core cycle later than otpu_tmem's (read latency 2: not the units' contract; for the
// out-of-context timing and the slot count, docs/tmem_pump.md).
//
// Behaviour, port for port, is otpu_tmem's with WPB = 1:
//   IMPL 0: a slot per read port's copy (the hosts: every port but the guests SH_MASK, which share
//           SH_HOST's slot in priority order), reads from the requests: 6 slots for the slice's 8
//           ports, 3 pumped copies instead of 6.
//   IMPL 1: banked, a slot per CPY value (NCP slots: the "copies" of otpu_tmem IMPL 1, two to a
//           pumped copy), a bank's slot address from the port granted there (r_en).
// The writes, the registered write stage, the bypass of a read that meets the landing write, and
// the lanes' data path are otpu_tmem's code. A write lands half a core cycle earlier than in
// otpu_tmem (the odd edge before the core edge), so a read at that core edge sees it in the RAM;
// the bypass returns the same word either way (it compares against the landing write, pw_*), so
// the read data are bit for bit otpu_tmem's. Reads (even edges) and writes (odd edges) never meet
// in one fast cycle, so the RAM's collision behaviour does not matter. And since the landing write
// is always in the RAM before the read, the bypass is not needed at all (BYP 0, the default: no
// pd registers, no address compare, no multiplexer between the RAM and the lanes; the testbench
// passes either way).
//
// The fast phase: per bank a toggle in the core domain (tog) and its copy in the fast domain (tq):
// they differ during the first half of a core cycle, so the next fast edge is odd; no reset (it
// aligns itself from the first core edge). Timing contract (rtl/exp/tmem_pump/ooc.tcl): the read
// address / enable inputs of a bank (u_pb ra / rena / rb / renb) are sampled on even fast edges
// only (a core-period multicycle path from clk); the RAM outputs change only after even edges
// (a core-period multicycle path into clk); the write stage and the toggle reach the RAM in one
// fast period.
module otpu_tmem_pump #(
  parameter int WORDS = 1 << 16,
  parameter int LANES = 8,
  parameter int NRP   = 8,
  parameter int NWP   = 4,
  parameter logic [NRP-1:0] GEN_R = '1,
  parameter logic [NWP-1:0] GEN_W = '1,
  parameter int SH_HOST = 0,
  parameter logic [NRP-1:0] SH_MASK = '0,
  parameter int IMPL  = 0,
  parameter int NCP   = 1,                 // IMPL 1: slots
  parameter logic [NRP-1:0][2:0] CPY = '0, // IMPL 1: each read port's slot (< NCP)
  parameter bit    RQ    = 1'b0,           // IMPL 1: slot addresses from the requests by PRANK
                                           // (otpu_tmem RQ; the slice's TMEM_RQ arbiter)
  parameter logic [NRP-1:0][2:0] PRANK = '0,
  parameter string STYLE = "block",        // the banks' ram_style ("block", "ultra")
  parameter bit    BYP   = 1'b0,           // keep otpu_tmem's bypass (redundant here: see below)
  parameter bit    FW    = 1'b1,           // the write stage and the phase in the fast domain
                                           // (below; 0: both in the core domain, as first built)
  parameter int    K     = 2,              // clk_f = K x clk: 2 (the copies above), or 3 / 4: one
                                           // copy of 2K - 1 slots, reads a cycle later (above)
  parameter int    CASC  = 0               // K > 2, block RAM: 1 keeps a bank's RAMs out of a
                                           // cascade (UltraScale+ cascade_height 1), 0 Vivado's
) (
  input  logic                                clk,
  input  logic                                clk_f,
  input  logic [NRP-1:0][LANES-1:0]           r_en,
  input  logic [NRP-1:0][LANES-1:0]           r_req,
  input  logic [NRP-1:0][LANES-1:0][31:0]     r_addr,
  output logic [NRP-1:0][LANES-1:0][31:0]     r_data,
  input  logic [NWP-1:0][LANES-1:0]           w_en,
  input  logic [NWP-1:0][LANES-1:0]           w_req,
  input  logic [NWP-1:0]                      w_gnt,
  input  logic [NWP-1:0][LANES-1:0][31:0]     w_addr,
  input  logic [NWP-1:0][LANES-1:0][31:0]     w_data
);
  localparam int BW = $clog2(LANES);
  localparam int BD = WORDS / LANES;
  localparam int IW = $clog2(BD);

  function automatic int host_of(input int p);
    if (IMPL == 0) return SH_MASK[p] ? SH_HOST : p;
    for (int q = 0; q < NRP; q++) if (CPY[q] == CPY[p]) return q;
    return p;
  endfunction
  // each host's slot: IMPL 0 the hosts in port order, IMPL 1 its CPY value
  function automatic int slot_of(input int h);
    int n;
    if (IMPL == 1) return int'(CPY[h]);
    n = 0;
    for (int q = 0; q < h; q++) if (host_of(q) == q) n++;
    return n;
  endfunction
  function automatic int n_slots();
    int n;
    if (IMPL == 1) return NCP;
    n = 0;
    for (int q = 0; q < NRP; q++) if (host_of(q) == q) n++;
    return n;
  endfunction
  // the host on slot s (-1: none)
  function automatic int host_on(input int s);
    for (int q = 0; q < NRP; q++) if (host_of(q) == q && slot_of(q) == s) return q;
    return -1;
  endfunction
  localparam int NS  = n_slots();
  localparam int NPC = (K > 2) ? 1 : (NS + 1) / 2;
  initial if (IMPL == 1) for (int p = 0; p < NRP; p++)
    if (CPY[p] >= NCP) $fatal(1, "otpu_tmem_pump: port %0d's slot %0d >= NCP %0d", p, CPY[p], NCP);
  initial if (K > 2 && (IMPL != 1 || NS != 2 * K - 1 || K > 4 || BYP))
    $fatal(1, "otpu_tmem_pump: K %0d needs IMPL 1, NCP %0d and no BYP", K, 2 * K - 1);

  // ---- write selection (otpu_tmem g_w1, WPB = 1)
  logic [LANES-1:0]         bw_v;
  logic [LANES-1:0][IW-1:0] bw_a;
  logic [LANES-1:0][31:0]   bw_d;
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
            bw_v[b] = 1'b1;
            bw_a[b] = bw_a[b] | pa[p][b];
            bw_d[b] = bw_d[b] | pdt[p][b];
          end
  end

  // ---- the registered write stage (otpu_tmem): lands at the next cycle's odd fast edge here.
  // FW: its registers are on clk_f, loaded at the even fast edges only (the core edges, where
  // otpu_tmem loads them): the same values at the same instants, but their inputs then cross from
  // the core domain in two fast periods (like the read addresses) and only one-level fast-domain
  // paths reach the RAMs. The even edges come from a phase of its own (pw_ev, as the banks').
  (* max_fanout = 16 *) logic [LANES-1:0]         pw_v;
  (* max_fanout = 16 *) logic [LANES-1:0][IW-1:0] pw_a;
  (* max_fanout = 16 *) logic [LANES-1:0][31:0]   pw_d;
  logic [LANES-1:0][31:0]                         pd;
  if (FW && K == 2) begin : g_fw
    (* keep *) logic wtog;
    (* keep *) logic wtq;
    (* max_fanout = 32 *) logic pw_ev;
    always_ff @(posedge clk) wtog <= !wtog;
    always_ff @(posedge clk_f) begin
      wtq <= wtog;
      pw_ev <= wtq != wtog;          // the next fast edge is even
    end
    always_ff @(posedge clk_f)
      if (pw_ev) begin
        pw_v <= bw_v;
        pw_a <= bw_a;
        pw_d <= bw_d;
      end
`ifndef SYNTHESIS
    initial begin wtog = 1'b0; wtq = 1'b0; pw_ev = 1'b1; end
`endif
  end else begin : g_sw
    always_ff @(posedge clk) begin
      pw_v <= bw_v;
      pw_a <= bw_a;
      pw_d <= bw_d;
    end
  end
  always_ff @(posedge clk)
    if (BYP && K == 2) pd <= pw_d;
`ifndef SYNTHESIS
  initial pw_v = '0;
`endif

  // ---- read ports' bank enables and rows (otpu_tmem)
  logic [NRP-1:0][LANES-1:0]         pb_en;
  logic [NRP-1:0][LANES-1:0][IW-1:0] pb_a;
  for (genvar p = 0; p < NRP; p++) begin : g_preq
    logic [LANES-1:0]         b_en;
    logic [LANES-1:0][IW-1:0] b_a;
    wire  [BW+IW-1:0]         a0 = r_addr[p][0][BW+IW-1:0];
    wire  [BW-1:0]            rot = a0[BW-1:0];
    wire  [LANES-1:0]         rs = (IMPL == 1 && !RQ) ? r_en[p] : r_req[p];
    always_comb begin
      b_en = '0; b_a = '0;
      if (GEN_R[p]) begin
        for (int b = 0; b < LANES; b++)
          for (int l = 0; l < LANES; l++)
            if (r_addr[p][l][BW-1:0] == BW'(b) && rs[l]) begin
              b_en[b] = 1'b1;
              b_a[b] = b_a[b] | r_addr[p][l][BW +: IW];
            end
      end else begin
        logic [LANES-1:0] t;
        t = '0;
        b_en = rs;
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

  // ---- each host's bank enables and rows (otpu_tmem's g_copy: IMPL 1 the granted port on its
  // slot, IMPL 0 the host or, at SH_HOST, the first guest that asks)
  logic [NRP-1:0][LANES-1:0]         h_en;
  logic [NRP-1:0][LANES-1:0][IW-1:0] h_a;
  for (genvar p = 0; p < NRP; p++) begin : g_host
    if (host_of(p) == p) begin : g_h
      if (IMPL == 1 && RQ) begin : g_bq
        always_comb begin
          h_en[p] = '0; h_a[p] = '0;
          for (int b = 0; b < LANES; b++)
            for (int r = NRP - 1; r >= 0; r--)
              for (int g = 0; g < NRP; g++)
                if (CPY[g] == CPY[p] && int'(PRANK[g]) == r && pb_en[g][b]) begin
                  h_en[p][b] = 1'b1;
                  h_a[p][b] = pb_a[g][b];
                end
        end
      end else if (IMPL == 1) begin : g_bk
        always_comb begin
          h_en[p] = '0; h_a[p] = '0;
          for (int g = 0; g < NRP; g++)
            if (CPY[g] == CPY[p])
              for (int b = 0; b < LANES; b++)
                if (pb_en[g][b]) begin
                  h_en[p][b] = 1'b1;
                  h_a[p][b] = h_a[p][b] | pb_a[g][b];
                end
        end
      end else if (p == SH_HOST && SH_MASK != '0) begin : g_sh
        always_comb begin
          logic take;
          h_en[p] = pb_en[p]; h_a[p] = pb_a[p];
          take = |r_req[p];
          for (int g = 0; g < NRP; g++)
            if (SH_MASK[g] && !take && |r_req[g]) begin
              h_en[p] = pb_en[g]; h_a[p] = pb_a[g]; take = 1'b1;
            end
        end
      end else begin : g_own
        assign h_en[p] = pb_en[p];
        assign h_a[p] = pb_a[p];
      end
    end else begin : g_g
      assign h_en[p] = '0;
      assign h_a[p] = '0;
    end
  end

  // ---- the pumped copies: copy c's bank b serves slot 2c on port A and slot 2c + 1 on port B
  logic [NRP-1:0][LANES-1:0][31:0] hq;     // each host's bank data
  if (K > 2) begin : g_k
    // one copy at K x: bank b's slot s is host_on(s)'s (otpu_pumpk_bank); its word comes a
    // cycle later than otpu_tmem's
    for (genvar b = 0; b < LANES; b++) begin : g_bank
      logic [2*K-2:1]         rens;
      logic [2*K-2:1][IW-1:0] ras;
      logic [2*K-2:0][31:0]   oq;
      for (genvar s = 1; s < 2 * K - 1; s++) begin : g_s
        localparam int HS = host_on(s);
        assign rens[s] = (HS >= 0) ? h_en[HS < 0 ? 0 : HS][b] : 1'b0;
        assign ras[s] = h_a[HS < 0 ? 0 : HS][b];
      end
      localparam int H0 = host_on(0);
      otpu_pumpk_bank #(.N(BD), .STYLE(STYLE), .K(K), .CASC(CASC)) u_pkb (
        .clk, .clk_f, .wv(pw_v[b]), .wa(pw_a[b]), .wd(pw_d[b]),
        .ren0((H0 >= 0) ? h_en[H0 < 0 ? 0 : H0][b] : 1'b0), .ra0(h_a[H0 < 0 ? 0 : H0][b]),
        .rens, .ras, .oq);
      for (genvar s = 0; s < 2 * K - 1; s++) begin : g_o
        localparam int HS = host_on(s);
        if (HS >= 0) begin : g_h
          assign hq[HS][b] = oq[s];
        end
      end
    end
  end else begin : g_k2
  for (genvar c = 0; c < NPC; c++) begin : g_pc
    localparam int HA = host_on(2 * c);
    localparam int HB = host_on(2 * c + 1);
    for (genvar b = 0; b < LANES; b++) begin : g_bank
      logic [31:0] qa, qb;
      otpu_pump_bank #(.N(BD), .STYLE(STYLE), .FW(FW)) u_pb (
        .clk, .clk_f, .rst(1'b0), .wv(pw_v[b]), .wa(pw_a[b]), .wd(pw_d[b]),
        .wv2(1'b0), .wa2('0), .wd2('0),
        .rena(HA >= 0 ? h_en[HA < 0 ? 0 : HA][b] : 1'b0), .ra(h_a[HA < 0 ? 0 : HA][b]),
        .renb(HB >= 0 ? h_en[HB < 0 ? 0 : HB][b] : 1'b0), .rb(h_a[HB < 0 ? 0 : HB][b]),
        .qa, .qb);
      if (HA >= 0) begin : g_a
        assign hq[HA][b] = qa;
      end
      if (HB >= 0) begin : g_b
        assign hq[HB][b] = qb;
      end
    end
  end
  end
  for (genvar p = 0; p < NRP; p++) begin : g_nohost
    if (host_of(p) != p) begin : g_z
      assign hq[p] = '0;
    end
  end

  // ---- per port: the bypass and the lanes' data path (otpu_tmem); K > 2 takes the requests a
  // cycle later, with the words
  logic [NRP-1:0][LANES-1:0]         ren_l;
  logic [NRP-1:0][LANES-1:0][BW-1:0] rbk_l;
  if (K > 2) begin : g_l2
    always_ff @(posedge clk)
      for (int p = 0; p < NRP; p++)
        for (int l = 0; l < LANES; l++) begin
          ren_l[p][l] <= r_en[p][l];
          rbk_l[p][l] <= r_addr[p][l][BW-1:0];
        end
`ifndef SYNTHESIS
    initial ren_l = '0;
`endif
  end else begin : g_l1
    always_comb
      for (int p = 0; p < NRP; p++)
        for (int l = 0; l < LANES; l++) begin
          ren_l[p][l] = r_en[p][l];
          rbk_l[p][l] = r_addr[p][l][BW-1:0];
        end
  end
  logic [NRP-1:0][LANES-1:0][31:0] pq;
  for (genvar p = 0; p < NRP; p++) begin : g_port
    if (host_of(p) == p) begin : g_copy
      logic [LANES-1:0][31:0] qb;
      for (genvar b = 0; b < LANES; b++) begin : g_bank
        if (BYP) begin : g_byp
          logic bh;
          always_ff @(posedge clk)
            if (h_en[p][b]) bh <= pw_v[b] && pw_a[b] == h_a[p][b];
          assign qb[b] = bh ? pd[b] : hq[p][b];
`ifndef SYNTHESIS
          initial bh = 1'b0;
`endif
        end else begin : g_nob
          assign qb[b] = hq[p][b];
        end
      end
      assign pq[p] = qb;
    end else begin : g_guest
      assign pq[p] = pq[host_of(p)];
    end
    wire  [BW-1:0]            rot = rbk_l[p][0];
    wire  [LANES-1:0][31:0]   qb = pq[p];
    logic [LANES-1:0][BW-1:0] sel;
    logic [LANES-1:0][BW-1:0] rq;
    logic [LANES-1:0]         fresh;
    logic [LANES-1:0][31:0]   held, lq;
    if (GEN_R[p]) begin : g_xb
      assign rq = '0;
    end else begin : g_rot
      (* keep *) logic [LANES-1:0][BW-1:0] rot_q;
      always_ff @(posedge clk)
        for (int l = 0; l < LANES; l++) if (|ren_l[p]) rot_q[l] <= rot;
      assign rq = rot_q;
    end
    always_ff @(posedge clk)
      for (int l = 0; l < LANES; l++) begin
        fresh[l] <= ren_l[p][l];
        if (GEN_R[p] && ren_l[p][l]) sel[l] <= rbk_l[p][l];
        if (fresh[l]) held[l] <= lq[l];
      end
    always_comb
      for (int l = 0; l < LANES; l++) begin
        lq[l] = GEN_R[p] ? qb[sel[l]] : qb[(l + int'(rq[l])) % LANES];
        r_data[p][l] = fresh[l] ? lq[l] : held[l];
      end
  end

`ifndef SYNTHESIS
  // IMPL 1: at most one granted port per slot and bank (the slice's arbiter, gnt_k; RQ: the
  // requests may meet, the arbiter grants the first)
  always @(posedge clk)
    if (IMPL == 1 && !RQ)
      for (int b = 0; b < LANES; b++)
        for (int p = 0; p < NRP; p++)
          for (int q = p + 1; q < NRP; q++)
            if (CPY[p] == CPY[q] && pb_en[p][b] && pb_en[q][b])
              $fatal(1, "otpu_tmem_pump: ports %0d and %0d granted in slot %0d bank %0d", p, q,
                     CPY[p], b);
`endif
endmodule

// otpu_pump_bank, one bank of a pumped copy at 2x: rtl/mem/otpu_tmem.sv (the one copy's
// pumped RAMs, otpu_tmem IMPL 3 PMP 1, build it on the boards)

// One bank of a copy pumped at K = 3 or 4 times the core clock (clk_f = K x clk, aligned): a true
// dual-port RAM serving 2K - 1 read slots and the write in each core cycle, with the requests
// registered, so a read's word comes in the cycle after next (a cycle later than otpu_tmem's). In
// the core cycle after a request, at fast edge e = 0 .. K-1 (e = 0 the core edge):
//   - port B reads slot e: slot 0 at the core edge from the request itself (ren0 / ra0, as the 2x
//     copies read), slots e >= 1 from the requests registered at that edge (sren / sra);
//   - port A writes the write stage at e = 0 (where otpu_tmem lands it), and reads slot K - 1 + e
//     at e >= 1.
// Each word is taken at the next fast edge (sq; the last edge's straight from the RAM) and all are
// presented at the next core edge (oq) for a whole core cycle. Slot 0 meets the landing write on
// the other port at e = 0: its word is the write stage's where the addresses match (bh / pd, as
// otpu_tmem's bypass); every later read sees that write in the RAM. The next edge's address and
// enable are registered a fast edge ahead (ad_*_r), so only a 2:1 multiplexer sits before the RAM.
// Timing contract (ooc.tcl): ren0 / ra0 / wv / wa / wd into the RAM, and the core-domain inputs
// of sren / sra / ad_*_r / en_*_r / oq, a core period (K fast periods); oq into the core domain a
// core period; the rest in one fast period.
(* keep_hierarchy = "yes" *)
module otpu_pumpk_bank #(
  parameter int    N     = 8192,
  parameter string STYLE = "block",
  parameter int    K     = 3,
  parameter int    CASC  = 0         // 1: the bank's block RAMs uncascaded (cascade_height 1): on
                                     // UltraScale+ Vivado splits 8K x 32 into cascaded RAMB36E2,
                                     // whose chain sits in the RAM-to-register path (one fast period)
) (
  input  logic                          clk,
  input  logic                          clk_f,
  input  logic                          wv,
  input  logic [$clog2(N)-1:0]          wa,
  input  logic [31:0]                   wd,
  input  logic                          ren0,
  input  logic [$clog2(N)-1:0]          ra0,
  input  logic [2*K-2:1]                rens,
  input  logic [2*K-2:1][$clog2(N)-1:0] ras,
  output logic [2*K-2:0][31:0]          oq
);
  localparam int NS = 2 * K - 1;
  localparam int AW = $clog2(N);
  // the phase: tog flips at every core edge, tq follows it a fast edge later and t2 one more; they
  // differ between fast edges 1 and 2 only, so ph (the fast cycle's index: e after edge e) is set
  // to 2 at edge 2 and counts otherwise (no reset: it aligns itself within the first core cycle)
  (* keep *) logic tog, tq, t2;
  logic [1:0] ph;
  always_ff @(posedge clk) tog <= !tog;
  always_ff @(posedge clk_f) begin
    tq <= tog;
    t2 <= tq;
    ph <= (tq != t2) ? 2'd2 : (ph == 2'(K - 1)) ? 2'd0 : ph + 2'd1;
  end
`ifndef SYNTHESIS
  initial begin tog = 1'b0; tq = 1'b0; t2 = 1'b0; ph = 2'd0; end
`endif
  wire ce = (ph == 2'(K - 1));             // this fast cycle ends at a core edge
  // the slots' requests, registered at the core edge
  logic [NS-1:1]         sren;
  logic [NS-1:1][AW-1:0] sra;
  always_ff @(posedge clk_f)
    if (ce) begin sren <= rens; sra <= ras; end
  // the next edge's reads, a fast edge ahead: at the edge after ph the next is n = ph + 2 (mod K);
  // n = 1 follows the core edge, whose registered requests are being loaded (from rens / ras)
  logic          ce_r, en_b_r, en_a_r;
  logic [AW-1:0] ad_b_r, ad_a_r;
  always_ff @(posedge clk_f) begin
    int n;
    n = (int'(ph) + 2) % K;
    ce_r <= (n == 0);
    if (ce) begin
      en_b_r <= rens[1]; ad_b_r <= ras[1];
      en_a_r <= rens[K]; ad_a_r <= ras[K];
    end else if (n != 0) begin
      en_b_r <= sren[n]; ad_b_r <= sra[n];
      en_a_r <= sren[K - 1 + n]; ad_a_r <= sra[K - 1 + n];
    end
  end
`ifndef SYNTHESIS
  initial ce_r = 1'b0;
`endif
  wire          en_b = ce_r ? ren0 : en_b_r;
  wire [AW-1:0] ad_b = ce_r ? ra0 : ad_b_r;
  wire          en_a = ce_r ? wv : en_a_r;
  wire          we_a = ce_r && wv;
  wire [AW-1:0] ad_a = ce_r ? wa : ad_a_r;
  logic [31:0] qa, qb;
  if (STYLE == "ultra") begin : g_u
    (* ram_style = "ultra" *) logic [31:0] mem [N];
    always_ff @(posedge clk_f)
      if (en_a) begin
        if (we_a) mem[ad_a] <= wd;
        else qa <= mem[ad_a];
      end
    always_ff @(posedge clk_f)
      if (en_b) qb <= mem[ad_b];
`ifndef SYNTHESIS
    initial for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else if (CASC == 1) begin : g_b1
    (* ram_style = "block", cascade_height = 1 *) logic [31:0] mem [N];
    always_ff @(posedge clk_f)
      if (en_a) begin
        if (we_a) mem[ad_a] <= wd;
        else qa <= mem[ad_a];
      end
    always_ff @(posedge clk_f)
      if (en_b) qb <= mem[ad_b];
`ifndef SYNTHESIS
    initial for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else begin : g_b
    (* ram_style = "block" *) logic [31:0] mem [N];
    always_ff @(posedge clk_f)
      if (en_a) begin
        if (we_a) mem[ad_a] <= wd;
        else qa <= mem[ad_a];
      end
    always_ff @(posedge clk_f)
      if (en_b) qb <= mem[ad_b];
`ifndef SYNTHESIS
    initial for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end
  // slot 0's bypass: the landing write (wv / wa / wd at the core edge) and slot 0's request
  logic        bh;
  logic [31:0] pd;
  always_ff @(posedge clk) begin
    bh <= wv && ren0 && wa == ra0;
    pd <= wd;
  end
  // the words: port B's slot e taken at edge e + 1, port A's slot K - 1 + e likewise; the last
  // edge's (slots K - 1 and NS - 1) go to oq straight from the RAM at the core edge
  logic [NS-1:0][31:0] sq;
  always_ff @(posedge clk_f) begin
    for (int e = 0; e < K - 1; e++) if (int'(ph) == e) sq[e] <= qb;
    for (int e = 1; e < K - 1; e++) if (int'(ph) == e) sq[K - 1 + e] <= qa;
    if (ce)
      for (int s = 0; s < NS; s++)
        oq[s] <= (s == 0) ? (bh ? pd : sq[0]) : (s == K - 1) ? qb : (s == NS - 1) ? qa : sq[s];
  end
endmodule
