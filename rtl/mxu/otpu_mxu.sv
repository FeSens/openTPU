// MXU: streams N rows x KB blocks of int8 from DRAM (port B, one D-byte chunk per cycle,
// prefetched through a FIFO) and their fp32 block scales (port A), and reuses each chunk across
// up to MCOLS stationary rows held in ACT RAM (docs/isa.md, MM). Per chunk and column j:
//   t[j][k] = (i2f(sum_i act[j][i] * w[i]) * ws) * ascale[j]
// 4-bit weights (flags[5:4] = WF: 1 int4, 2 E2M1): a D-byte chunk holds two blocks (the low
// half first) and is consumed in two advances; each block's scale word is a bf16 scale (ws) and
// four 4-bit multipliers m_b of the sub-block sums: sum_i -> sum_b m_b * sum_{i in b} (exact).
//   acc[j]  = isum_4(t[j][0..KB-1])      interleaved partials p[k mod 4], then (p0+p2)+(p1+p3)
// PAIR (flags[6], 4-bit, 2M <= MCOLS; docs/isa.md "Column reuse"): a chunk is consumed in one
// advance. Columns j < M take its low block 2c (ACT block ab+2c), columns j+M its high block
// 2c+1 (ACT block ab+2c+1); the two scale words arrive together (port A's 8-byte pair), and
//   acc[j]  = isum_4(t[j][2c] + t[j+M][2c+1])      (+0 for the missing block 2c+1 of an odd KB)
// After the last block of a row the M results are written to TMEM (optionally accumulated,
// optionally rescaled: y = old * alpha[j] + acc). With RMAX the running max of every written
// value per column is written after the last row (the max order is total: any order is exact).
// Replay (ROWS > MCOLS, M > MCOLS): each streamed row is consumed G = ceil(M / MCOLS) times,
// once per group of MCOLS stationary rows (ACT RAM rows g*MCOLS ..). Group 0 pops the row's
// chunks from the FIFO without freeing them, the later groups read the same FIFO entries again
// (head + k) and the last group frees them, so a weight is streamed from DRAM once for all M
// rows (a row must fit the FIFO: KB <= DEPTH). Each (row, group) is a row of the pipeline and
// the drain; the drain steps its TMEM addresses by MCOLS * ors per group.
//
// Pipelined for the FPGA clock:
//   pop | operands | products | pairs (2) | +4 | tree | i2f (2) | *ws (2) | *ascale (2) | [pair (4)]
//   | partial loop (4) | combine
// (pair: PAIR only, t + the partner column's t; other MMs skip its four stages: a command's
// blocks never share the pipeline with another command's, so the latency may differ by mode)
// The partial loop is exactly 4 pipeline advances long (one 4-stage adder), so block k meets
// the partial of block k-4 of the same row. The compute pipeline
// advances when a chunk is popped, or with a bubble whenever the next chunk would start a new
// row (or the command has no chunks left); it only freezes in the middle of a row.
// Finished rows go to a result FIFO; the drain writes up to LANES results per cycle, through a
// pipelined read-modify-write for ACC/ASCALE, and holds while the TMEM grant is withheld.
//
// The MXU holds two commands: the issuer streams the newer one's chunks while the consumer
// finishes (and drains) the older one. The issuer yields DRAM ports to the DMA and QST
// (a_gnt/b_gnt). On the FPGA the products map to DSP48 multipliers.
module otpu_mxu
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int D     = 32,
  parameter int MCOLS = 8,
  parameter int ROWS  = MCOLS,   // ACT RAM rows: the most stationary rows of a command
  parameter int DEPTH = 16,
  parameter int LANES = 8,       // TMEM banks (MCOLS > LANES drains a row in several cycles)
  parameter int IMPL  = 0,       // integer dot product: 0 adder tree, 1 DSP cascade chains,
                                 // 2 systolic array (docs/mxu_systolic.md)
  parameter int CL    = 16,      // IMPL 1, 2: products per cascade chain (D / CL chains)
  parameter int SID   = 0
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  start,       // accept: the issuer may stream it
  input  logic                  go,          // release: the consumer may use it (in order)
  input  cmd_t                  cmd,
  output logic                  rdy,
  output logic                  done,
  output logic                  computing,   // a chunk is consumed this cycle (profiling)
  // profiling: this cycle's stream state (the FIFO level; work but no chunk / chunks but no
  // consumption) and, a cycle after a command ends, its counters {deny, frz, bp, starve}
  output logic [$clog2(DEPTH):0] pf_level,
  output logic                  pf_starve,
  output logic                  pf_block,
  output logic                  pf_u,
  output logic [3:0][31:0]      pf_uv,
  // ACT RAM read
  output logic [15:0]           act_blk,
  output logic [15:0]           act_blk2,    // PAIR: the odd block, read by the rows in act_hi
  output logic [MCOLS-1:0]      act_hi,
  output logic [7:0]            act_grp,
  output logic                  act_ren,     // the ACT RAM read register advances (with S0)
  input  logic [MCOLS*D*8-1:0]  act_data,
  input  logic [MCOLS*32-1:0]   act_scale,
  // DRAM port A (scales) and B (chunks)
  output logic                  a_req,
  output logic [31:0]           a_addr,
  input  logic                  a_gnt,
  input  logic                  a_rvalid,
  input  logic [31:0]           a_rdata,
  input  logic [31:0]           a_rdata2,    // the other word of a_rdata's 8-byte pair
  output logic                  b_req,
  output logic [31:0]           b_addr,
  input  logic                  b_gnt,
  input  logic                  b_rvalid,
  input  logic [D*8-1:0]        b_rdata,
  // TMEM (read port for ACC, write port)
  output logic [LANES-1:0]        t_ren,
  output logic [LANES-1:0][31:0]  t_raddr,
  input  logic [LANES-1:0][31:0]  t_rdata,
  output logic [LANES-1:0]        t_wen,
  output logic [LANES-1:0][31:0]  t_waddr,
  output logic [LANES-1:0][31:0]  t_wdata,
  input  logic                    t_gnt
);
  localparam int BW = $clog2(LANES);
  localparam int PW = $clog2(DEPTH);
  localparam int LM = 2, LA = 4;
  localparam int NPART = 4;                 // MM partials (isum_4)
  // result FIFO rows: a row pops only while fewer than RF are between the pop and the drain, so
  // RF covers the pipeline's rows in flight (one-block rows: one per cycle of its latency); IMPL 2's
  // longer dot product needs 64
  localparam int RF = (IMPL == 2) ? 64 : 32;
  localparam int RFW = $clog2(RF);
  localparam int MW = $clog2(MCOLS) + 1;
  localparam int NL = (LANES < MCOLS) ? LANES : MCOLS;   // lanes the drain can fill

  // conflict-free drain run for a row stride: LANES / gcd(ors, LANES), at most NL
  function automatic logic [MW-1:0] drain_run(input logic [15:0] ors);
    int d;
    d = 1;                                   // ors a multiple of LANES: one lane per cycle
    for (int t = BW - 1; t >= 0; t--)        // the lowest set bit t: gcd = 2^t
      if (ors[t]) d = LANES >> t;
    return MW'((d < NL) ? d : NL);
  endfunction
  initial if (D % 16 != 0) $fatal(1, "otpu_mxu: D must be a multiple of 16");
  localparam logic [1:0] WF_W8 = 2'd0, WF_W4F = 2'd2;
  localparam int SDEPTH = 2 * DEPTH;         // scale entries: up to two per chunk (4-bit)
  localparam int NP = (MCOLS + 1) / 2;       // DSP pairs (columns 2p, 2p+1 share a multiplier)
  localparam int SPW = $clog2(SDEPTH);
  initial if (ROWS < MCOLS || ROWS > 255) $fatal(1, "otpu_mxu: MCOLS <= ROWS < 256");
`ifndef SYNTHESIS
  // a replayed row (M > MCOLS) must fit the FIFO: its chunks (4-bit: two blocks each)
  always @(posedge clk) if (!rst && start) begin
    if (int'(cmd.w6[23:16]) > ROWS) $fatal(1, "otpu_mxu: M %0d > ROWS %0d", cmd.w6[23:16], ROWS);
    if (int'(cmd.w6[23:16]) > MCOLS && ((cmd.flags[5:4] != WF_W8) ?
        (int'(cmd.w4[31:16]) + 1) / 2 : int'(cmd.w4[31:16])) > DEPTH)
      $fatal(1, "otpu_mxu: a replayed row (M > MCOLS) must fit the FIFO: KB %0d, DEPTH %0d",
             cmd.w4[31:16], DEPTH);
    if (int'(cmd.w6[23:16]) > MCOLS && cmd.flags[6])
      $fatal(1, "otpu_mxu: PAIR needs 2*M <= MCOLS");
  end
`endif

  // ================================================================== issuer
  logic        i_act, i_unit, i_w4, i_pair;
  logic [31:0] i_left, i_rs, i_srs;
  logic [15:0] i_KB, i_k;
  logic [31:0] row_addr, chunk_addr, srow_addr, scale_addr;
  logic [PW:0] occ;                         // chunks issued and not yet popped (<= DEPTH)

  // ================================================================== command queue (2)
  // PAIR: 4-bit only; a row is ceil(KB/2) advances of one chunk each
  wire        cmd_pair = cmd.flags[6] && cmd.flags[5:4] != WF_W8;
  wire [15:0] cmd_KBa = cmd_pair ? (cmd.w4[31:16] + 16'd1) >> 1 : cmd.w4[31:16];
  wire [31:0] cmd_total = 32'(cmd.w4[15:0]) * 32'(cmd_KBa);   // advances (chunk requests)
  // cmd_total == 0 without the multiply (a product of two 16-bit factors is zero iff one is;
  // cmd_KBa is zero iff KB is, or PAIR's 16-bit KB + 1 wraps): off the DSP's pattern detect
  wire        cmd_tz = (cmd.w4[15:0] == 16'd0) || (cmd.w4[31:16] == 16'd0) ||
                       (cmd_pair && (&cmd.w4[31:16]));
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && start && cmd_tz != (cmd_total == 0))
      $fatal(1, "otpu_mxu: cmd_tz %0d but total %0d", cmd_tz, cmd_total);
`endif
  // The two entries are the head (h) and the next command (n), not two slots and a head pointer:
  // every head field is its own flop, with no 2:1 mux on a pointer in front of the consumer and
  // the drain. Completing the head swaps them (n keeps the old head, as the idle slot did).
  typedef struct packed {
    logic [31:0]      out, total;
    logic             tz;              // total == 0 (registered: off the drain path)
    logic [15:0]      KB, KBa;         // KBa: advances per row
    logic             pair;
    logic [MCOLS-1:0] hi;              // PAIR: the columns that take the odd blocks (j >= M)
    logic [31:0]      mxo;             // RMAX output base: out + M * ors (no multiply later)
    logic [7:0]       M, ab;
    logic [MW-1:0]    run;             // drain lanes per cycle without a bank conflict
    logic             unit, acc, rmax, asc, go;
    logic [1:0]       wf;
    logic [31:0]      asa;
    logic [MCOLS-1:0][31:0] jo;        // j * ors
    logic [7:0]       G;               // groups: ceil(M / MCOLS)
    logic [MW-1:0]    Ml;              // rows of the last group
    logic [31:0]      gs;              // drain address step to the next group (MCOLS * ors)
    logic             g1;              // G == 1: a row's first group is its last
    logic [MW-1:0]    dnl;             // the last group's first drain step: its lanes (min(run,
    logic             dfl;             // Ml)) and whether it drains the group (Ml <= run)
  } qent_t;
  qent_t       h, n;                    // the head, the next command
  qent_t       cmd_e;                   // cmd as an entry (not yet released)
  logic [1:0]  q_n;

  always_comb begin
    int g;
    cmd_e = '0;
    cmd_e.out   = cmd.w3;
    cmd_e.total = cmd_total;
    cmd_e.tz    = cmd_tz;
    cmd_e.KB    = cmd.w4[31:16];
    cmd_e.KBa   = cmd_KBa;
    cmd_e.pair  = cmd_pair;
    for (int j = 0; j < MCOLS; j++) cmd_e.hi[j] = cmd_pair && 32'(j) >= 32'(cmd.w6[23:16]);
    cmd_e.mxo   = cmd.w3 + 32'(cmd.w6[23:16]) * 32'(cmd.w6[15:0]);
    cmd_e.M     = cmd.w6[23:16];
    cmd_e.run   = drain_run(cmd.w6[15:0]);
    cmd_e.ab    = cmd.w6[31:24];
    cmd_e.unit  = cmd.flags[0];
    cmd_e.wf    = cmd.flags[5:4];
    cmd_e.acc   = cmd.flags[1];
    cmd_e.rmax  = cmd.flags[2];
    cmd_e.asc   = cmd.flags[3];
    cmd_e.go    = 1'b0;
    cmd_e.asa   = cmd.w2;
    for (int j = 0; j < MCOLS; j++) cmd_e.jo[j] = 32'(j) * 32'(cmd.w6[15:0]);
    g = (int'(cmd.w6[23:16]) + MCOLS - 1) / MCOLS;
    if (g < 1) g = 1;
    cmd_e.G     = 8'(g);
    cmd_e.Ml    = MW'(int'(cmd.w6[23:16]) - (g - 1) * MCOLS);
    cmd_e.gs    = 32'(MCOLS) * 32'(cmd.w6[15:0]);
    cmd_e.g1    = (g == 1);
    cmd_e.dnl   = (cmd_e.run < cmd_e.Ml) ? cmd_e.run : cmd_e.Ml;
    cmd_e.dfl   = (cmd_e.Ml <= cmd_e.run);
  end

  // Two heads (IMPL 2: commands overlap in the pipeline; docs/mxu_systolic.md). The drain's head
  // is h, the oldest command: its results are drained, its completion signalled. The pop head is
  // h, or n once the head's chunks have all popped (pn), if the pipeline treats n alike (the same
  // PAIR and M, the only per-command state read after S0): the next command's rows then enter the
  // pipeline behind the head's instead of after its drain. pop_q is the pop head's entry, a register
  // loaded with h and n (pn ? n : h), so neither the pop nor the drain reads its command through a
  // mux. OVL = 0 (IMPL 0 / 1): the pop head is h.
  localparam bit OVL = (IMPL == 2);
  logic        pn;
  qent_t       pop_q, pop_e;
  assign pop_e = OVL ? pop_q : h;

  wire [31:0] c_out = h.out;
  wire        c_tz = h.tz;
  // the pop head's (c_KB .. c_wf, p_*) and the drain head's (the rest)
  wire [15:0] c_KB = pop_e.KB, c_KBa = pop_e.KBa;
  wire        c_pair = pop_e.pair;
  wire [MCOLS-1:0] c_hi = pop_e.hi;
  wire [7:0]  c_M = h.M, c_ab = pop_e.ab;
  wire [7:0]  p_M = pop_e.M, p_G = pop_e.G;
  wire        p_act = (q_n != 0) && pop_e.go;
  wire [MW-1:0] c_run = h.run;
  wire        c_unit = pop_e.unit, c_acc = h.acc, c_rmax = h.rmax;
  wire        c_asc = h.asc;
  wire [1:0]  c_wf = pop_e.wf;
  wire        c_w4 = (c_wf != WF_W8);
  wire [31:0] c_asa = h.asa;
  wire        c_act = (q_n != 0) && h.go;
  wire [7:0]  c_G = h.G;
  logic [31:0] alpha [MCOLS];
  logic [1:0]  al_st;                       // ASCALE factors: 0 to load, 1 loading, 2 loaded
  logic [7:0]  al_i, mx_i;                  // next ASCALE factor to load / RMAX value to write

  // FIFOs of chunks and of their scales (the two DRAM ports return independently, in order;
  // an issued chunk's slot is reserved, so neither FIFO can overflow)
  // block RAM: written in their own reset-free process below (a write under the control
  // process's reset made Vivado build them from ~22K LUTs of distributed RAM, with a write
  // address fanning out to every LUT). The chunk FIFO is its own module (kept as a hierarchy):
  // inline, Vivado absorbed its read register into the DSP input registers of the products,
  // which left an asynchronous read, and built it from 5,472 RAM64M anyway.
  (* ram_style = "block" *) logic [63:0]    f_scale [SDEPTH];   // {a_rdata2, a_rdata}
  logic [PW-1:0]  f_head, f_tail;
  logic [SPW-1:0] s_head, s_tail;
  logic [PW:0]    f_count;
  logic [SPW:0]   s_count;
  // FIFO read addresses, kept equal to (last group ? head : head + entry k of the row) by the
  // pops: a register, so no adder sits in front of the block RAM address
  logic [PW-1:0]  f_rd;
  logic [SPW-1:0] s_rd;
  // The chunk FIFO is FG block RAM column groups (whole 36-bit BRAM columns each, so no more
  // block RAMs than one RAM of the chunk's width), each with its own copies of the write and
  // read addresses (f_tl, f_rdl: f_tail's and f_rd's updates): one register no longer drives
  // every block RAM of the chunk across its span (133.33 MHz, bb7f844: f_rd -> u_fd ADDRB, 0
  // levels, fanout 31, 6.6 ns of route, +0.081 ns, the core's worst; 110ec6d: f_tail -> ADDRA
  // +0.201). f_tail and f_rd remain for the checks.
  localparam int FW = D * 8;
  localparam int FC = (FW + 35) / 36;       // 36-bit block RAM columns
  localparam int FG = (FC < 4) ? FC : 4;
  function automatic int fg_lo(input int g);
    int c;
    c = 0;
    for (int i = 0; i < g; i++) c = c + FC / FG + ((i < FC % FG) ? 1 : 0);
    return (36 * c < FW) ? 36 * c : FW;
  endfunction
  (* keep *) logic [FG-1:0][PW-1:0] f_tl, f_rdl;

  // ================================================================== consumer control
  logic [15:0] ck;
  logic [7:0]  cg;                          // the group of the row being consumed (replay)
  logic [31:0] c_left;                      // chunks of the pop head not yet freed
  // a command accepted last cycle became the head: c_left takes its total now, from h (a flop),
  // not from cmd_total (the sequencer's command through the 16 x 16 multiply) in the start cycle.
  // It cannot pop yet (released a cycle after its start at the earliest), and nothing may treat
  // the stale c_left == 0 as drained meanwhile
  logic        cl_ld;
  logic [RFW:0] rows_live;                  // the head's rows popped (first block), not drained
  logic [RFW:0] rows_p;                     // the pop head's, while it is not the head (OVL)
  wire last_k   = (ck + 1 == c_KBa);
  wire last_g   = (cg + 1 == p_G);          // the row's last group: its pops free FIFO entries
  wire more     = p_act && (c_left != 0);
  // the chunk of advance ck (4-bit without PAIR: two advances a chunk) and whether the advance
  // finishes it
  wire [15:0] ckc = (c_w4 && !c_pair) ? {1'b0, ck[15:1]} : ck;
  wire cdone    = !c_w4 || c_pair || ck[0] || last_k;
  // advance ck's chunk and scale are in the FIFO: the next entries (one group), or entry ck of
  // the row (group 0 of several; the later groups find the whole row)
  wire f_av     = (p_G == 8'd1) ? (f_count != 0) : (cg != 0 || 32'(f_count) > 32'(ckc));
  wire s_av     = (p_G == 8'd1) ? (s_count != 0) : (cg != 0 || 32'(s_count) > 32'(ck));
  // pop: one block (PAIR: one chunk) advances into the pipeline; fpop: its chunk leaves the
  // FIFO (4-bit without PAIR: after the high half, or after the row's last block; replay: in the
  // row's last group)
  // IMPL 2 (docs/mxu_systolic.md): a row's first advance pops only once all of the row's chunks
  // and scales are in the FIFOs (a later group finds them there), so a row never stalls in its
  // middle and the compute pipeline needs no clock enable: en_c is 1, and a cycle without a pop
  // is a bubble between rows, as at every row start today
  wire [15:0] c_rch = (c_w4 && !c_pair) ? 16'((32'(c_KBa) + 1) / 2) : c_KBa;   // chunks a row
  wire row_in   = (IMPL != 2) || cg != 0 ||
                  (32'(f_count) >= 32'(c_rch) && (c_unit || 32'(s_count) >= 32'(c_KBa)));
  wire pop      = more && f_av && (c_unit || s_av) &&
                  (ck != 0 || (row_in && (OVL ? (RFW+2)'(rows_live) + (RFW+2)'(rows_p) < (RFW+2)'(RF)
                                              : rows_live < RF)));
  wire fpop     = pop && last_g && cdone;
  wire en_c     = (IMPL == 2) || pop || !(more && ck != 0);   // freeze only in the middle of a row
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && IMPL == 2) begin
    if (more && ck != 0 && !pop) $fatal(1, "otpu_mxu: IMPL 2 row stalled at advance %0d", ck);
    if (more && 32'(c_rch) > DEPTH) $fatal(1, "otpu_mxu: IMPL 2 needs a row (%0d chunks) to fit the FIFO", c_rch);
  end
  always @(posedge clk) if (!rst && more) begin
    if (f_rd != (last_g ? f_head : f_head + PW'(ckc)))
      $fatal(1, "otpu_mxu: FIFO read address %0d, head %0d, k %0d", f_rd, f_head, ck);
    if (!c_unit && s_rd != (last_g ? s_head : s_head + SPW'(ck)))
      $fatal(1, "otpu_mxu: scale read address %0d, head %0d, k %0d", s_rd, s_head, ck);
  end
`endif
  // the issuer walks advances: a chunk request with every 8-bit block, every even 4-bit block
  // (an odd one's chunk is already on its way, so it needs no FIFO slot) and every PAIR chunk; a
  // scale request with each (PAIR: the pair of words of blocks 2c, 2c+1)
  wire need_b   = !i_w4 || i_pair || !i_k[0];
  wire want_iss = i_act && (!need_b || occ < (PW+1)'(DEPTH));
  wire go_iss   = want_iss && (!need_b || b_gnt) && (i_unit || a_gnt);

  assign rdy = !i_act && (q_n < 2);
  assign computing = pop;
  assign pf_level  = f_count;
  assign pf_starve = more && f_count == 0;
  assign pf_block  = more && f_count != 0 && !pop;
  assign b_req  = go_iss && need_b;
  // not gated with b_req: the address (and the slice's parity of it) does not wait for b_gnt
  assign b_addr = chunk_addr >> 2;
  assign a_req  = go_iss && !i_unit;
  assign a_addr = a_req ? (scale_addr >> 2) : '0;
  assign act_blk = 16'(c_ab) + (c_pair ? {ck[14:0], 1'b0} : ck);
  assign act_blk2 = 16'(c_ab) + {ck[14:0], 1'b1};
  assign act_hi = c_hi;
  assign act_grp = cg;
  assign act_ren = en_c;

  // ================================================================== compute pipeline
  typedef struct packed {
    logic       v;
    logic       last;     // last block of its row
    logic       first;    // block index < 4: the partial starts at +0
    logic [1:0] q;        // block index mod 4
    logic       h;        // 4-bit: the chunk's high half
    logic       o;        // PAIR: the chunk's high block is in the row (not past an odd KB)
  } cm_t;

  cm_t                 m0, m1, m2, m3, m4, m5, m6;
  logic [D*8-1:0]      w0;                  // the chunk (FIFO read register)
  logic [15:0]         mb0, mb0h;           // S0's sub-block multipliers m_3..m_0 (8-bit: 1;
                                            // h: PAIR's high block)
  logic [MCOLS*D*8-1:0] a0;
  f32_t                ws0, ws1, ws2, ws3, ws4, ws5;
  f32_t                ws0h, ws1h, ws2h, ws3h, ws4h, ws5h;
  f32_t                ws6 [MCOLS];
  f32_t                as0 [MCOLS];
  f32_t                fi [MCOLS];
  // The dot product of D int8 x int8 products is exact in SW bits: |dot| <= D*2^14 < 2^MG.
  localparam int SW = 16 + $clog2(D);
  localparam int MG = SW - 1;               // magnitude bits
  localparam int LZW = $clog2(MG);
  logic signed [SW-1:0] s4 [MCOLS];
  // i2f of a dot product (MG <= 24): the magnitude converts exactly, so there is no rounding
  // step; the same result as i2f(32'(x)). S5: sign, magnitude, leading zeros | S6: normalize.
  typedef struct packed {
    logic           z, s;
    logic [LZW-1:0] lz;
    logic [MG-1:0]  mag;
  } dmid_t;
  function automatic dmid_t d2f_s1(input logic signed [SW-1:0] x);
    dmid_t m;
    m.z   = (x == 0);
    m.s   = x[SW-1];
    m.mag = MG'(x[SW-1] ? -x : x);                        // |x| < 2^MG
    m.lz  = LZW'(lzc32(32'(m.mag) << (32 - MG)));        // leading zeros within MG bits
    return m;
  endfunction
  function automatic f32_t d2f_s2(input dmid_t m);
    logic [MG-1:0] nrm;
    if (m.z) return F_ZERO;
    nrm = m.mag << m.lz;                                  // nrm[MG-1] = 1
    return {m.s, 8'(126 + MG - int'(m.lz)), 23'(nrm[MG-2:0]) << (24 - MG)};
  endfunction
  // S0's ACT RAM block and scales are the ACT RAM's registered read
  assign a0 = act_data;
  // the scale FIFO's registered read is kept free of logic (so it maps into the block RAM's
  // output register); unit-scale chunks are substituted after it
  logic [63:0] ws0r;
  logic cu0, pr0;
  logic [1:0] wf0;
  logic [MCOLS-1:0] hi0;                    // PAIR: the columns of the high block
  wire  w40 = (wf0 != WF_W8);
  assign ws0 = cu0 ? F_ONE : w40 ? {ws0r[15:0], 16'd0} : ws0r[31:0];   // 4-bit: bf16 scale
  assign mb0 = (cu0 || !w40) ? 16'h1111 : ws0r[31:16];
  assign ws0h = cu0 ? F_ONE : {ws0r[47:32], 16'd0};
  assign mb0h = cu0 ? 16'h1111 : ws0r[63:48];
  // 4-bit elements: nibble i of the chunk half; int4 two's complement, E2M1 as twice its value
  // ({0, 1, 2, 3, 4, 6, 8, 12}, sign in bit 3)
  function automatic logic [7:0] dec4(input logic [3:0] c, input logic fp);
    if (!fp) return {{4{c[3]}}, c};
    case (c)
      4'd5: return 8'd6;
      4'd6: return 8'd8;
      4'd7: return 8'd12;
      4'd8: return 8'd0;
      4'd9: return -8'sd1;
      4'd10: return -8'sd2;
      4'd11: return -8'sd3;
      4'd12: return -8'sd4;
      4'd13: return -8'sd6;
      4'd14: return -8'sd8;
      4'd15: return -8'sd12;
      default: return 8'(c);
    endcase
  endfunction
  // the weight of position i for a column that takes the high block (PAIR) or the high half
  function automatic logic [7:0] wsel(input logic [D*8-1:0] w, input int i, input logic hb,
                                      input logic [1:0] wf);
    if (wf == WF_W8) return w[i*8 +: 8];
    return dec4(w[(hb ? D*4 : 0) + 4*i +: 4], wf == WF_W4F);
  endfunction
  always_comb for (int j = 0; j < MCOLS; j++) as0[j] = act_scale[j*32 +: 32];

  // FIFO writes (the slot was reserved when the chunk was issued; see occ)
  always_ff @(posedge clk) begin
    if (a_rvalid) f_scale[s_tail] <= {a_rdata2, a_rdata};
  end
  for (genvar g = 0; g < FG; g++) begin : g_fd
    localparam int LO = fg_lo(g);
    localparam int W = ((g == FG - 1) ? FW : fg_lo(g + 1)) - LO;
    (* keep_hierarchy = "yes" *)
    otpu_ram_sdp #(.W(W), .N(DEPTH)) u_fd (
      .clk, .we(b_rvalid), .wa(f_tl[g]), .wd(b_rdata[LO +: W]), .re(en_c), .ra(f_rdl[g]),
      .rd(w0[LO +: W]));
  end
`ifndef SYNTHESIS
  // the groups' address copies are f_tail and f_rd (all reset together)
  bit fd_rst;                                         // (registers start arbitrary)
  initial fd_rst = 1'b0;
  always @(posedge clk) begin
    if (rst) fd_rst <= 1'b1;
    for (int g = 0; g < FG; g++)
      if (!rst && fd_rst && (f_tl[g] != f_tail || f_rdl[g] != f_rd))
        $fatal(1, "otpu_mxu: FIFO group %0d addresses %0d / %0d, not %0d / %0d", g, f_tl[g],
               f_rdl[g], f_tail, f_rd);
  end
`endif

  always_ff @(posedge clk) if (en_c) begin
    // S0: the popped chunk, its ACT RAM block and scales
    m0 <= '0;
    if (pop) begin
      m0.v <= 1'b1;
      m0.last <= last_k;
      m0.first <= (ck < 16'(NPART));
      m0.q <= ck[1:0];
      m0.h <= c_w4 && !c_pair && ck[0];
      m0.o <= c_pair && !(last_k && c_KB[0]);
    end
    ws0r <= f_scale[s_rd];
    cu0 <= c_unit;
    wf0 <= c_wf;
    pr0 <= c_pair;
    hi0 <= c_hi;
    m5 <= m4; ws5 <= ws4; ws5h <= ws4h;
    m6 <= m5;
  end
  // S5, S6: int -> fp32
  if (MG <= 24) begin : g_d2f
    dmid_t im [MCOLS];
    always_ff @(posedge clk) if (en_c) begin
      for (int j = 0; j < MCOLS; j++) im[j] <= d2f_s1(s4[j]);
      for (int j = 0; j < MCOLS; j++) fi[j] <= d2f_s2(im[j]);
    end
  end else begin : g_i2f
    i2f_mid_t im [MCOLS];
    always_ff @(posedge clk) if (en_c) begin
      for (int j = 0; j < MCOLS; j++) im[j] <= i2f_s1(32'(s4[j]));
      for (int j = 0; j < MCOLS; j++) fi[j] <= i2f_s2(im[j]);
    end
  end
  // ws6 feeds the first fp multiplier's B operand: a reset flop, never an SRL tap (per column:
  // PAIR's high-block columns take the high block's scale)
  always_ff @(posedge clk)
    for (int j = 0; j < MCOLS; j++)
      if (rst) ws6[j] <= '0; else if (en_c) ws6[j] <= hi0[j] ? ws5h : ws5;

  // dot-product latency S0 -> s4 (the tree: 7 register levels: operands (the decoded weights),
  // products, pairs (two: the DSP cascade), groups, sub-blocks times their multipliers, block sum)
  localparam int NG = D / CL;
  localparam int TL = (NG <= 1) ? 0 : (NG <= 4) ? 1 : (NG <= 16) ? 2 : 3;
  localparam int CLS = (D / 4 < CL) ? D / 4 : CL;    // IMPL 2's chains: within a 4-bit sub-block
  localparam int LDOT = (IMPL == 0) ? 7 : (IMPL == 2) ? CLS + MCOLS + 3
                                          : CL + 1 + TL - (TL >= 3 ? 1 : 0);

  // ---- S1 .. S4: the exact integer dot products of the chunk with every column's ACT block;
  // s4 (with m4, ws4) is the chunk's result LDOT cycles after S0.
  if (IMPL == 0) begin : g_tree
    // products, pair sums, then an 8 / D/16 adder tree (one register level each)
    // Columns 2p and 2p+1 share the weight byte, so one multiplier (a DSP48 with its pre-adder)
    // makes both: pm = (a0*2^16 + a1) * w = (a0*w)*2^16 + a1*w (a 25-bit A: shift 17 overflows).
    // The DSP post-adders sum positions 2q and 2q+1 in a systolic pair: DSP 2q makes M + PK (PREG),
    // DSP 2q+1 takes its operands a stage later (pre-adder register ADREG, BREG = 2) and adds that
    // through its cascade input (M + PCIN, PREG), so the sum needs no fabric adder or register
    // (a one-cycle M + M + PK was built in fabric: 44 flip-flops and 11 CARRY4s per pair sum):
    // pq = E*2^16 + (O + PK), E / O the pair sums of columns 2p / 2p+1, both in [-32512, 32768].
    // O + PK is in [1, 65281], so the fields need no borrow: E = pq[32:16] (signed) and
    // O + PK = pq[15:0] (unsigned); column 2p+1's group sums start at -(GS/2)*PK. PK is odd
    // (no constant trailing zeros to trim from the post-adder). An odd last column keeps plain
    // products, paired in fabric.
    // A PAIR split between the pair's columns (M = 2p+1: column 2p takes the low block, 2p+1 the
    // high one, different weights) packs both 4-bit weights into the multiplier's other operand,
    // B = wl*2^13 + wh (|w| <= 12: 18 bits), so the one product holds both columns' products:
    //   pm = a0*wl*2^29 + (a0*wh*2^16 + a1*wl*2^13) + a1*wh
    // The cross terms sit between them, |a0*wh*2^16 + a1*wl*2^13| < 2^27 per position. With
    // PK2 = 2^28 + 2^12 the pair sum's low 29 bits are in (0, 2^29) and its low field in
    // [2^12 - 3072, 2^12 + 3072]: E = pq[43:29] (signed), O + 2^12 = pq[12:0] (unsigned).
    // pm, pq and pr are packed so they are registers the DSPs absorb (MREG, PREG), not memories
    // that are mapped to fabric flops after DSP packing; signed fields are read through $signed().
    localparam int NDP = MCOLS / 2;                     // DSP pairs
    localparam logic [43:0] PK = 44'd32513, PK2 = 44'h000_1000_1000;
    logic [NP-1:0][D/2-1:0][43:0] pme, pmo;             // positions 2q / 2q+1: M registers
    logic [NP-1:0][D/2-1:0][43:0] pe;                   // DSP 2q's P: M + PK (to 2q+1's PCIN)
    logic [NP-1:0][D/2-1:0][24:0] pao;                  // DSP 2q+1's pre-adder output (ADREG)
    logic [NP-1:0][D/2-1:0][17:0] wro;                  // DSP 2q+1's B, a stage later (BREG 2)
    logic [NP-1:0][D/2-1:0][43:0] pq;
    logic [D-1:0][15:0]           pr;
    logic [D/2-1:0][16:0]         prq, prq2;
    // group sums of GS positions (16; D/4 when smaller), GPB groups per 4-bit sub-block
    localparam int GS = (D / 4 < 16) ? D / 4 : 16;
    localparam int NG3 = D / GS, GPB = NG3 / 4;
    logic signed [19:0] s3 [MCOLS][NG3];
    logic [SW-1:0] v [MCOLS][4];
    logic [15:0] mbz, mb1, mbc, mb2, mb3, mbzh, mb1h, mbch, mb2h, mb3h;
    cm_t  mz, mc, mt4;
    f32_t wz, wc, wt4, wzh, wch, wt4h;
    // the operands registered once more (the multipliers' input registers): the weight decode
    // sits between the chunk FIFO's read register and here, not in front of the multipliers
    logic [MCOLS*D*8-1:0]       ar;
    logic [NP-1:0][D-1:0][17:0] wr;                    // per DSP pair (B)
    logic [D-1:0][7:0]          wrl;                   // an odd last column's weights
    logic [NP-1:0]              sp;                    // the pair is split (with the operands)
    logic [NP-1:0][43:0]        pkr;
    always_ff @(posedge clk) if (en_c) begin
      ar <= a0;
      for (int p = 0; p < NDP; p++) begin
        logic split, hb;
        split = pr0 && !hi0[2*p] && hi0[2*p+1];
        hb = pr0 ? hi0[2*p] : m0.h;
        // B = wl*2^13 + wh: wh sign-extended to 13 bits, above it wl - (wh < 0); else sext(w)
        for (int i = 0; i < D; i++) begin
          logic [7:0] lo8, b8;
          b8 = wsel(w0, i, split || hb, wf0);
          lo8 = wsel(w0, i, 1'b0, wf0);
          wr[p][i] <= {split ? 5'(lo8[4:0] - 5'(b8[7])) : {5{b8[7]}}, {5{b8[7]}}, b8};
        end
        sp[p] <= split;
        pkr[p] <= split ? PK2 : PK;
      end
      if (MCOLS % 2 == 1)
        for (int i = 0; i < D; i++) wrl[i] <= wsel(w0, i, pr0 ? hi0[MCOLS-1] : m0.h, wf0);
      for (int p = 0; p < NDP; p++)
        for (int q = 0; q < D / 2; q++) begin
          logic signed [24:0] pa;
          pa = $signed({ar[(2*p*D + 2*q)*8 +: 8], 16'b0}) + 25'($signed(ar[((2*p+1)*D + 2*q)*8 +: 8]));
          pme[p][q] <= 44'(pa) * 44'($signed(wr[p][2*q]));
          pe[p][q] <= pme[p][q] + pkr[p];
          pao[p][q] <= $signed({ar[(2*p*D + 2*q+1)*8 +: 8], 16'b0}) +
                       25'($signed(ar[((2*p+1)*D + 2*q+1)*8 +: 8]));
          wro[p][q] <= wr[p][2*q+1];
          pmo[p][q] <= 44'($signed(pao[p][q])) * 44'($signed(wro[p][q]));
          pq[p][q] <= pmo[p][q] + pe[p][q];
        end
      if (MCOLS % 2 == 1) begin
        for (int i = 0; i < D; i++)
          pr[i] <= 16'(int'($signed(ar[((MCOLS-1)*D + i)*8 +: 8])) * int'($signed(wrl[i])));
        for (int q = 0; q < D / 2; q++)
          prq[q] <= 17'($signed(pr[2*q])) + 17'($signed(pr[2*q+1]));
        prq2 <= prq;                                   // with the pairs' second stage
      end
      for (int j = 0; j < MCOLS; j++) begin
        for (int g = 0; g < NG3; g++) begin
          // column 2p+1's pair fields carry +PK (split: +2^12) each: the group sum starts at
          // -(GS/2) times that
          logic signed [19:0] t;
          t = (j % 2 == 0) ? '0 : sp[j/2] ? 20'(-(GS / 2 * 4096)) : 20'(-(GS / 2 * int'(PK)));
          for (int k = 0; k < GS / 2; k++)
            if (j % 2 == 1)
              t = t + 20'(sp[j/2] ? {3'b0, pq[j/2][GS/2*g+k][12:0]} : pq[j/2][GS/2*g+k][15:0]);
            else if (j + 1 < MCOLS)
              t = t + (sp[j/2] ? 20'($signed(pq[j/2][GS/2*g+k][43:29]))
                               : 20'($signed(pq[j/2][GS/2*g+k][32:16])));
            else t = t + 20'($signed(prq2[GS/2*g+k]));
          s3[j][g] <= t;
        end
        // sub-block sums times their multipliers (a DSP pre-adder and multiplier), then the
        // block sum; exact in SW bits (|sum| < 2^(SW-1) in every format)
        for (int b = 0; b < 4; b++) begin
          logic [SW-1:0] t;
          t = '0;
          for (int g = 0; g < GPB; g++) t = t + SW'(s3[j][b*GPB + g]);
          v[j][b] <= SW'(t * SW'(hi0[j] ? mb3h[4*b +: 4] : mb3[4*b +: 4]));
        end
        begin
          logic [SW-1:0] t;
          t = '0;
          for (int b = 0; b < 4; b++) t = t + v[j][b];
          s4[j] <= $signed(t);
        end
      end
      mz <= m0; wz <= ws0; mbz <= mb0; wzh <= ws0h; mbzh <= mb0h;
      m1 <= mz; ws1 <= wz; mb1 <= mbz; ws1h <= wzh; mb1h <= mbzh;
      mc <= m1; wc <= ws1; mbc <= mb1; wch <= ws1h; mbch <= mb1h;
      m2 <= mc; ws2 <= wc; mb2 <= mbc; ws2h <= wch; mb2h <= mbch;
      m3 <= m2; ws3 <= ws2; mb3 <= mb2; ws3h <= ws2h; mb3h <= mb2h;
      mt4 <= m3; wt4 <= ws3; wt4h <= ws3h;
      m4 <= mt4; ws4 <= wt4; ws4h <= wt4h;
    end
  end else if (IMPL == 1) begin : g_casc
    // Systolic accumulate chains (DSP48 A*B + PCIN cascades): the D positions form NG = D / CL
    // chains of CL; position i = g*CL + k enters stage k of chain g k cycles after S0 (operand
    // skew in shift registers), so a new chunk enters every cycle. Stage k: a registered product
    // (the DSP's M register) added to stage k-1's running sum (its P register). The NG chain
    // sums meet in a small adder tree (TL register levels of up to 4 inputs). Exact integers.
    initial if (D % CL != 0 || NG > 64) $fatal(1, "otpu_mxu: CL must divide D, D/CL <= 64");
    // operand skew: position k of every chain is delayed k cycles (weights shared by columns)
    logic [7:0] ws_k [D];                               // skewed weight bytes
    logic [7:0] as_k [MCOLS][D];                        // skewed activation bytes
`ifndef SYNTHESIS
    always_ff @(posedge clk) if (pop && c_w4) $fatal(1, "otpu_mxu: 4-bit weights need IMPL = 0");
`endif
    for (genvar i = 0; i < D; i++) begin : g_wsk
      otpu_delay #(.W(8), .N(i % CL)) u_w (.clk, .en(en_c), .d(w0[i*8 +: 8]), .q(ws_k[i]));
      for (genvar j = 0; j < MCOLS; j++) begin : g_ask
        otpu_delay #(.W(8), .N(i % CL)) u_a (.clk, .en(en_c), .d(a0[(j*D + i)*8 +: 8]),
                                            .q(as_k[j][i]));
      end
    end
    // the chains
    logic signed [15:0] mreg [MCOLS][D];                // stage products (M registers)
    logic signed [23:0] preg [MCOLS][D];                // running sums (P registers)
    always_ff @(posedge clk) if (en_c) begin
      for (int j = 0; j < MCOLS; j++)
        for (int i = 0; i < D; i++) begin
          mreg[j][i] <= 16'(int'($signed(as_k[j][i])) * int'($signed(ws_k[i])));
          preg[j][i] <= ((i % CL == 0) ? 24'sd0 : preg[j][i - 1]) + 24'(mreg[j][i]);
        end
    end
    // chain sums (stage CL-1 of each chain) -> tree
    logic signed [31:0] t1 [MCOLS][(NG + 3) / 4];
    logic signed [31:0] t2 [MCOLS][(NG + 15) / 16];
    always_ff @(posedge clk) if (en_c) begin
      for (int j = 0; j < MCOLS; j++) begin
        for (int u = 0; u < (NG + 3) / 4; u++) begin
          logic signed [31:0] t;
          t = '0;
          for (int v = 0; v < 4; v++)
            if (4 * u + v < NG) t = t + 32'(preg[j][(4 * u + v) * CL + CL - 1]);
          t1[j][u] <= t;
        end
        for (int u = 0; u < (NG + 15) / 16; u++) begin
          logic signed [31:0] t;
          t = '0;
          for (int v = 0; v < 4; v++)
            if (4 * u + v < (NG + 3) / 4) t = t + t1[j][4 * u + v];
          t2[j][u] <= t;
        end
      end
    end
    always_comb
      for (int j = 0; j < MCOLS; j++) begin
        logic signed [31:0] t;
        t = '0;
        case (TL)
          0: t = 32'(preg[j][CL - 1]);
          1: t = t1[j][0];
          2: t = t2[j][0];
          default: for (int u = 0; u < (NG + 15) / 16; u++) t = t + t2[j][u];
        endcase
        s4[j] = SW'(t);
      end
    // the meta and the weight scale travel alongside (S0 -> S4 position)
    otpu_delay #(.W($bits(cm_t)), .N(LDOT)) u_m4 (.clk, .en(en_c), .d(m0), .q(m4));
    otpu_delay #(.W(32), .N(LDOT)) u_ws4 (.clk, .en(en_c), .d(ws0), .q(ws4));
    otpu_delay #(.W(32), .N(LDOT)) u_ws4h (.clk, .en(en_c), .d(ws0h), .q(ws4h));
  end else begin : g_sys
    // A 2D systolic array (docs/mxu_systolic.md). The chunk is decoded once (S1), position i is
    // delayed by its chain stage k = i % CL, and the weights then move one register hop per column:
    // column j sees them j cycles after column 0 (fan-out 2, no broadcast). Both streams flow: the
    // low block (or the 4-bit half of a non-PAIR advance) and PAIR's high block; each column takes
    // one in its DSPs' pre-adder (A = low, D = high, INMODE by its per-command hi0[j], a copy in
    // each DSP's INMODE register: no fabric select). Column j's activation byte i is delayed k + j
    // (shift registers) into the DSP's B register. Along D: chains of CL products, one per DSP48E1
    // (otpu_pe: M register, P = PCIN + M). The 4-bit sub-blocks are two chains each, so the
    // sub-block multipliers apply at the chain ends, as in IMPL 0; column j's s4 is then delayed
    // MCOLS - 1 - j cycles, and all columns reach the epilogue together (LDOT = CLS + MCOLS + 3).
    // Exact integers: bit-identical to IMPL 0. Chains of CLS = min(CL, D/4) positions, so a
    // sub-block (D/4 positions) is CPS whole chains.
    localparam int CPS = D / 4 / CLS;
    initial if ((D / 4) % CLS != 0) $fatal(1, "otpu_mxu: IMPL 2 needs CL to divide D/4");
    logic [D-1:0][15:0] wd;                             // S1: {high block, low block / half}
    always_ff @(posedge clk) if (en_c)
      for (int i = 0; i < D; i++) wd[i] <= {wsel(w0, i, 1'b1, wf0), wsel(w0, i, m0.h, wf0)};
    logic [D-1:0][15:0] wc [MCOLS];                    // the streams at column j
    for (genvar i = 0; i < D; i++) begin : g_wsk
      otpu_skew #(.W(16), .N(i % CLS)) u_w (.clk, .en(en_c), .d(wd[i]), .q(wc[0][i]));
    end
    always_ff @(posedge clk) if (en_c)
      for (int j = 1; j < MCOLS; j++) wc[j] <= wc[j-1];
    logic [7:0] as_k [MCOLS][D];
    logic signed [23:0] preg [MCOLS][D];                // running sums (the DSPs' P registers;
                                                        // a chain's last: its P, unregistered)
    logic [47:0] pc [MCOLS][D];                         // their cascade outputs
`ifndef SYNTHESIS
    bit hi_v;                                           // hi0 and the PEs' selects are loaded
    initial hi_v = 1'b0;
    always @(posedge clk) if (en_c) hi_v <= 1'b1;
`endif
    for (genvar j = 0; j < MCOLS; j++) begin : g_ask
      for (genvar i = 0; i < D; i++) begin : g_p
        otpu_skew #(.W(8), .N(i % CLS + j)) u_a (.clk, .en(en_c), .d(a0[(j*D + i)*8 +: 8]),
                                               .q(as_k[j][i]));
        // columns past the first register their weights in the DSP (WREG), from the previous
        // column's hop: a copy of wc[j] (133.33 MHz, 110ec6d: wc -> PE A / D, 0 levels, 95%
        // route, +0.266 ns, 36 endpoints)
        localparam int JW = (j == 0) ? 0 : j - 1;
        otpu_pe #(.FIRST(i % CLS == 0), .LAST(i % CLS == CLS - 1), .WREG(j != 0)) u_pe (
          .clk, .en(en_c), .act(as_k[j][i]), .wlo(wc[JW][i][7:0]), .whi(wc[JW][i][15:8]),
          .sel(c_hi[j]),
          .pcin((i % CLS == 0) ? 48'd0 : pc[j][(i % CLS == 0) ? i : i - 1]), .pcout(pc[j][i]),
          .p(preg[j][i]));
`ifndef SYNTHESIS
        // the DSP's select register is hi0[j] (both are c_hi[j], loaded with en_c)
        always @(posedge clk) begin
          if (hi_v && u_pe.selr != hi0[j])
            $fatal(1, "otpu_mxu: PE %0d.%0d's select %0d is not hi0 %0d", j, i, u_pe.selr, hi0[j]);
          // and its weight registers are wc[j] (both wc[j - 1], loaded with en_c)
          if (hi_v && j != 0 && {u_pe.whr, u_pe.wlr} != wc[j][i])
            $fatal(1, "otpu_mxu: PE %0d.%0d's weights are not wc", j, i);
        end
`endif
      end
    end
    // chain ends (S0 + CLS + 2 + j, in otpu_colend's A / D registers): the sub-block sums times
    // their multipliers (v, S0 + CLS + 3 + j), then the block sum (s4r, + 4 + j), in the column's
    // four otpu_colend DSPs; the multipliers of column j (its block's under PAIR) travel
    // alongside, into their B registers. A sub-block's chain ends: the first half on the
    // pre-adder's A, the rest on D
    localparam int CPA = (CPS + 1) / 2;
    logic signed [SW-1:0] s4r [MCOLS];
    for (genvar j = 0; j < MCOLS; j++) begin : g_col_end
      logic [15:0] mbd;
      logic [3:0][23:0] ca, cd;
      otpu_skew #(.W(16), .N(CLS + 1 + j)) u_mb (.clk, .en(en_c), .d(hi0[j] ? mb0h : mb0),
                                               .q(mbd));
      always_comb
        for (int b = 0; b < 4; b++) begin
          ca[b] = '0; cd[b] = '0;
          for (int c = 0; c < CPS; c++)
            if (c < CPA) ca[b] = ca[b] + preg[j][(b*CPS + c)*CLS + CLS - 1];
            else cd[b] = cd[b] + preg[j][(b*CPS + c)*CLS + CLS - 1];
        end
      otpu_colend #(.SW(SW)) u_ce (.clk, .en(en_c), .a(ca), .d(cd), .mb(mbd), .s(s4r[j]));
      otpu_skew #(.W(SW), .N(MCOLS - 1 - j)) u_dsk (.clk, .en(en_c), .d(s4r[j]), .q(s4[j]));
    end
    otpu_delay #(.W($bits(cm_t)), .N(LDOT)) u_m4 (.clk, .en(en_c), .d(m0), .q(m4));
    otpu_delay #(.W(32), .N(LDOT)) u_ws4 (.clk, .en(en_c), .d(ws0), .q(ws4));
    otpu_delay #(.W(32), .N(LDOT)) u_ws4h (.clk, .en(en_c), .d(ws0h), .q(ws4h));
  end

  // the ACT scale travels with the chunk to the second multiplier (S0 + LDOT + 2 + LM)
  f32_t t1 [MCOLS], t2 [MCOLS], tq [MCOLS], pacc [MCOLS];
  cm_t  mt_p, mt, mq_p, mq, mx, ma;          // meta at the pair adder input and output, the loop
                                             // adder input (mq under PAIR, else mt), its output
  logic [7:0] M0;                            // the command's M (constant while its blocks flow)
  always_ff @(posedge clk) if (en_c) M0 <= p_M;
  // the delay lines into the fp operands end in reset flops (a reset can't go into an SRL, so the
  // last stage is an FDRE with a fast clock-to-out); same total length and enable
  otpu_delay #(.W($bits(cm_t)), .N(2 * LM - 1)) u_mt (.clk, .en(en_c), .d(m6), .q(mt_p));
  always_ff @(posedge clk) if (rst) mt <= '0; else if (en_c) mt <= mt_p;
  otpu_delay #(.W($bits(cm_t)), .N(LA - 1)) u_mq (.clk, .en(en_c), .d(mt), .q(mq_p));
  always_ff @(posedge clk) if (rst) mq <= '0; else if (en_c) mq <= mq_p;
  assign mx = pr0 ? mq : mt;
  otpu_delay #(.W($bits(cm_t)), .N(LA)) u_ma (.clk, .en(en_c), .d(mx), .q(ma));
  for (genvar j = 0; j < MCOLS; j++) begin : g_col
    f32_t fb, prev, as_p, asq, tx;
    otpu_delay #(.W(32), .N(LDOT + 2 + LM - 1)) u_as (.clk, .en(en_c), .d(as0[j]), .q(as_p));
    always_ff @(posedge clk) if (rst) asq <= '0; else if (en_c) asq <= as_p;
    otpu_fmul #(.LAT(LM)) u_m1 (.clk, .en(en_c), .a(fi[j]), .b(ws6[j]), .y(t1[j]));
    otpu_fmul #(.LAT(LM)) u_m2 (.clk, .en(en_c), .a(t1[j]), .b(asq), .y(t2[j]));
    // pair (PAIR): a column that takes a low block (j < M <= MCOLS/2) adds its partner's term,
    // column j + M's, when the high block is in the row, else +0; the other columns' results are
    // not used under PAIR
    if (j < MCOLS / 2) begin : g_pair
      f32_t pb;
      always_comb begin
        pb = F_ZERO;
        for (int m = 1; m <= MCOLS / 2; m++)
          if (j + m < MCOLS && mt.o && M0 == 8'(m)) pb = t2[j + m];
      end
      otpu_fadd #(.LAT(LA)) u_pr (.clk, .en(en_c), .a(t2[j]), .b(pb), .y(tq[j]));
      assign tx = pr0 ? tq[j] : t2[j];
    end else begin : g_nopair
      assign tq[j] = '0;
      assign tx = t2[j];
    end
    // partial loop: pacc(block k) = pacc(block k - 4) + t(k), exactly NPART advances
    otpu_delay #(.W(32), .N(NPART - LA)) u_fb (.clk, .en(en_c), .d(pacc[j]), .q(fb));
    assign prev = mx.first ? F_ZERO : fb;
    otpu_fadd #(.LAT(LA)) u_acc (.clk, .en(en_c), .a(prev), .b(tx), .y(pacc[j]));
  end

  // combine a row's final partials: (p0+p2)+(p1+p3). Block k's partial meets its isum_4 partner,
  // block k-2 (two advances earlier, same row: a row's blocks are consecutive advances); blocks
  // k < 2 meet +0 (a slot the row never fills). One pair completes at block KB-2, the other at
  // KB-1: consecutive advances, so one adder makes both, and the earlier sum waits a step in cy1
  // (+0 when KB = 1: that pair is 0+0). The order within a pair may swap; fp_add commutes.
  wire zb = !ma.v || ma.last || (ma.first && ma.q == 2'd0);   // the next block has k < 2
  logic mid_y;                                                 // the block now at cy: not last
  otpu_delay #(.W(1), .N(LA)) u_my (.clk, .en(en_c), .d(ma.v && !ma.last), .q(mid_y));
  f32_t pp1 [MCOLS], pb [MCOLS], cy [MCOLS], cy1 [MCOLS], rowv [MCOLS];
  always_ff @(posedge clk) if (en_c)
    for (int j = 0; j < MCOLS; j++) begin
      pp1[j] <= pacc[j];                          // block t
      pb[j]  <= zb ? F_ZERO : pp1[j];             // block t-1: the partner of block t+1
      cy1[j] <= mid_y ? cy[j] : F_ZERO;           // the earlier pair sum
    end
  wire launch = ma.v && ma.last;
  logic lv2;
  for (genvar j = 0; j < MCOLS; j++) begin : g_comb
    otpu_fadd #(.LAT(LA)) u_p (.clk, .en(en_c), .a(pacc[j]), .b(pb[j]), .y(cy[j]));
    otpu_fadd #(.LAT(LA)) u_c (.clk, .en(en_c), .a(cy1[j]), .b(cy[j]), .y(rowv[j]));
  end
  // the two adder levels, and the row registered before the result FIFO (the second adder's
  // output ran into the FIFO's LUT RAM in one cycle: 0.19 ns slack at 125.49 MHz)
  f32_t rowr [MCOLS];
  always_ff @(posedge clk) if (en_c) for (int j = 0; j < MCOLS; j++) rowr[j] <= rowv[j];
  otpu_delay #(.W(1), .N(2 * LA + 1)) u_lv (.clk, .en(en_c), .d(launch), .q(lv2));

  // ================================================================== result FIFO
  // Rows in LUT RAM (was flip-flops); the drain reads the head row from flip-flops. rf_q trails
  // the head by the last cycle's pop (rf_hs: the head is rf_q + rf_hs), so the rows are read at
  // rf_q, rf_q + 1 and rf_q + 2 (a row written this cycle bypassed) with no grant on the address;
  // each cycle the head row and the row after it are registered (rh0, rh1) and the last pop picks
  // the head (rfh). The grant only reaches rf_hs, and the drain's lanes start at a 2:1 mux instead
  // of a 64:1 one on rf_h (133.33 MHz, 61e015ff4: rf_h -> 64:1 -> t_wdata -> TMEM pw_d, 10
  // levels, 89% route, -0.486 ns; the pop -> rf_h replicas' enable, -0.492). One RAM copy per
  // read address: a single read port each infers as LUT RAM (three ports on one array do not).
  logic [RFW-1:0] rf_q, rf_t;
  logic [RFW:0]   rf_n;
  logic           rf_hs;                     // the last cycle popped a row
  wire         rf_push = en_c && lv2;
  logic [MCOLS-1:0][31:0] rowp, rh0, rh1, rfh;
  logic [2:0][MCOLS-1:0][31:0] rq;           // rows rf_q + c, as they are after this cycle
  logic [2:0][RFW-1:0]    rf_qc;
  always_comb begin
    for (int j = 0; j < MCOLS; j++) rowp[j] = rowr[j];
    for (int c = 0; c < 3; c++) rf_qc[c] = rf_q + RFW'(c);
  end
  for (genvar c = 0; c < 3; c++) begin : g_rfm
    (* ram_style = "distributed" *) logic [MCOLS*32-1:0] m [RF];   // (flat: a row of MCOLS words)
    always_ff @(posedge clk) if (!rst && rf_push) m[rf_t] <= rowp;
    assign rq[c] = (rf_push && rf_t == rf_qc[c]) ? rowp : m[rf_qc[c]];
  end
  always_ff @(posedge clk) begin
    rh0 <= rf_hs ? rq[1] : rq[0];           // the head (rf_q + rf_hs) as it is after this cycle
    rh1 <= rf_hs ? rq[2] : rq[1];           // the row after it
  end
  assign rfh = rf_hs ? rh1 : rh0;

  // ================================================================== drain
  // lanes this cycle: results dj .. dj+ncnt-1 of the head row, stopping at a bank conflict
  logic [31:0] drow [MCOLS];                 // replay: dad of the head row's group 0
  logic [31:0] dad [MCOLS];                  // head row's TMEM addresses: out + n + j * ors,
                                             // kept incrementally (no adder between the drain's
                                             // lane pick and the arbiter)
  // dj, ncnt: MW bits (0 < M <= MCOLS < 2^MW, which the ISA requires), so the lane test and the
  // row-done compare are a few bits wide instead of carry chains
  logic [MW-1:0] dj;
  logic [MW-1:0] ncnt;
  logic [7:0]    dg;                         // the group of the row being drained
  logic [7:0]    dg1;                        // dg + 1, kept alongside (no adder before the
                                             // last-group compare: BL16 at 120.755 MHz had
                                             // dg -> dg + 1 == c_G -> drain lanes -> TMEM grant
                                             // -> mk / dad / drow enables, 16 levels, -0.287 ns)
  // The drain's step is registered, with its next value computed a cycle ahead (below, at the
  // head's entries): no compare or subtract between the head's entry and the TMEM write request,
  // which feeds the TMEM arbiter, every unit's grant and the VPU's write-buffer enable (133.33 MHz:
  // h.G -> dg1 == c_G -> c_Mn - dj -> min(run, .) -> lanes -> MXU write mask -> VPU grant -> en_q,
  // 15 levels, -0.446 ns; the same into TMEM's pw_* and the MXU's mk enables). d_last: dg1 ==
  // c_G; ncnt = min(run, c_Mn - dj) and dln its lanes (k < ncnt); d_fin: the step drains the
  // group's last results (dj + ncnt == c_Mn). drain_go's counts are tested by registered flags
  // (rf_nz: rf_n != 0, rl_nz: rows_live != 0) for the same reason.
  logic          d_last, d_fin, rf_nz, rl_nz;
  logic [NL-1:0] dln;
  wire  [MW-1:0] c_Mn = d_last ? h.Ml : MW'(MCOLS);   // its results
  logic [LANES-1:0][31:0] daddr_l, dval_l;
  // The drain lanes' addresses, dad[dj + k], are registered (dal): set wherever dj or dad move,
  // the step's from registers (dad[dj + ncnt + k]), so the grant only picks. The TMEM write
  // request and the arbiter's bank masks then start at flip-flops, not at the dj multiplexer
  // (133.33 MHz, 110ec6d: dj -> dad[dj + k] -> t_waddr -> the MXU's bank mask -> the grants ->
  // the VPU's WBUF head enables, 12 levels, +0.168 ns; -> TMEM's pw_a, +0.274)
  logic [NL-1:0][31:0] dal;
  function automatic logic [NL-1:0][31:0] dlane_a(input logic [31:0] a [MCOLS],
                                                  input logic [MW-1:0] j);
    for (int k = 0; k < NL; k++) dlane_a[k] = (32'(j) + 32'(k) < MCOLS) ? a[MW'(j) + MW'(k)] : '0;
  endfunction
  logic [LANES-1:0][7:0]  dcol_l;
  // (OVL: the result FIFO's head row is the head command's while it has rows not yet drained)
  wire  drain_go = rf_nz && (!OVL || rl_nz) && (!c_asc || al_st == 2'd2);
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && drain_go && dg1 != dg + 8'd1) $fatal(1, "otpu_mxu: dg1 %0d, dg %0d", dg1, dg);
`endif
  // Results j and j' of a row sit ors * (j' - j) words apart: the same bank iff that is a multiple
  // of LANES. So consecutive results are conflict-free in runs of LANES / gcd(ors, LANES)
  // (capped at NL), a per-command constant: the lane count is min(run, M - dj), with no serial
  // bank check between the result FIFO and the TMEM request.
  // A step of a group with `left` results still to drain: {drains them all, lanes}
  function automatic logic [MW:0] dstep(input logic [MW-1:0] left, input logic [MW-1:0] run);
    return {left <= run, (run < left) ? run : left};
  endfunction
  function automatic logic [NL-1:0] dlanes(input logic [MW-1:0] cnt);
    for (int k = 0; k < NL; k++) dlanes[k] = 32'(k) < 32'(cnt);
  endfunction
`ifndef SYNTHESIS
  // the registered step is its definition from the head's entry, dg1, dj and the counts
  always @(posedge clk) if (!rst && q_n != 0) begin
    logic [MW-1:0] mn, lf;
    mn = (dg1 == c_G) ? h.Ml : MW'(MCOLS);
    lf = mn - dj;
    if (d_last != (dg1 == c_G) || ncnt != ((c_run < lf) ? c_run : lf) || dln != dlanes(ncnt) ||
        d_fin != (MW'(dj + ncnt) == mn))
      $fatal(1, "otpu_mxu: drain step d_last %0d ncnt %0d d_fin %0d (dg1 %0d G %0d dj %0d M %0d run %0d)",
             d_last, ncnt, d_fin, dg1, c_G, dj, mn, c_run);
  end
  always @(posedge clk) if (!rst && q_n != 0 && dal != dlane_a(dad, dj))
    $fatal(1, "otpu_mxu: the drain lanes' addresses are not dad[dj + k] (dj %0d)", dj);
  always @(posedge clk) if (!rst && (rf_nz != (rf_n != 0) || rl_nz != (rows_live != 0)))
    $fatal(1, "otpu_mxu: rf_nz %0d (rf_n %0d), rl_nz %0d (rows_live %0d)", rf_nz, rf_n, rl_nz, rows_live);
`endif
  // (the lanes' addresses and values do not wait for the count: only the enables do)
  always_comb begin
    daddr_l = '0; dval_l = '0; dcol_l = '0;
    daddr_l[NL-1:0] = dal;
    for (int k = 0; k < NL; k++)
      if (32'(dj) + 32'(k) < MCOLS) begin
        dval_l[k] = rfh[MW'(dj) + MW'(k)];
        dcol_l[k] = 8'(dj) + 8'(k);
      end
  end
`ifndef SYNTHESIS
  // PAIR reads a block's scale word and its partner's as one 8-byte aligned pair
  always @(posedge clk) begin
    if (a_req && i_pair && scale_addr[2]) $fatal(1, "otpu_mxu: PAIR scales must be 8-byte aligned");
    if (start && cmd_pair && 2 * int'(cmd.w6[23:16]) > MCOLS)
      $fatal(1, "otpu_mxu: PAIR needs 2 * M <= MCOLS");
  end
  // the lane count is what a greedy pick that stops at the first bank conflict would take
  always @(posedge clk) if (drain_go) begin
    logic [LANES-1:0] used;
    int n;
    used = '0; n = 0;
    for (int k = 0; k < NL && 32'(dj) + 32'(k) < 32'(c_Mn); k++) begin
      if (used[dad[MW'(dj) + MW'(k)][BW-1:0]]) break;
      used[dad[MW'(dj) + MW'(k)][BW-1:0]] = 1'b1;
      n++;
    end
    if (n != 32'(ncnt)) $fatal(1, "otpu_mxu: drain lanes %0d, greedy pick %0d", ncnt, n);
  end
`endif
  wire drain_row_done = drain_go && d_fin;   // dj + ncnt == c_Mn
  // the next drain step's candidates, from registers (the grant only picks one): the rest of the
  // group, or (d_fin) the next group, the row's last (d_nl) or not; a new head's is its entry's
  wire [MW-1:0]    d_rest = c_Mn - dj - ncnt;
  wire             d_nl = d_last ? h.g1 : (dg1 + 8'd1 == c_G);
  wire [MW:0]      st_mid = dstep(d_rest, c_run);
  wire [MW:0]      st_grp = d_nl ? {h.dfl, h.dnl} : dstep(MW'(MCOLS), c_run);
  wire [NL-1:0]    ln_mid = dlanes(st_mid[MW-1:0]), ln_grp = dlanes(st_grp[MW-1:0]);
  wire             d_pop = t_gnt && drain_row_done;   // a row leaves the result FIFO
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && rf_n != 0 && rfh != g_rfm[0].m[RFW'(rf_q + RFW'(rf_hs))])
    $fatal(1, "otpu_mxu: registered head row is not row %0d", rf_q + RFW'(rf_hs));
`endif
  // RMAX lanes (mx_i + k < c_M), registered as the drain's; next: mx_i + LANES. Their TMEM
  // addresses h.mxo + mx_i + k are registered too (mxl, loaded from h.mxo while no RMAX write
  // runs: mx_go is never set in a head's first cycle), so no adder precedes the write request
  // and the arbiter's bank masks (61e015ff4: h.mxo -> two carry chains -> amk -> MXU grant ->
  // rf_h enables, 8 levels, -0.492 ns)
  logic [LANES-1:0] mxm, mxm_nx;
  logic [LANES-1:0][31:0] mxl;
  always_comb for (int k = 0; k < LANES; k++) mxm_nx[k] = 32'(mx_i) + LANES + 32'(k) < 32'(c_M);
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && q_n != 0)
    for (int k = 0; k < LANES; k++)
      if (mxm[k] != (32'(mx_i) + 32'(k) < 32'(c_M)))
        $fatal(1, "otpu_mxu: RMAX lane %0d: mxm %0d, mx_i %0d, M %0d", k, mxm[k], mx_i, c_M);
  always @(posedge clk) if (!rst && mx_go)
    for (int k = 0; k < LANES; k++)
      if (mxl[k] != h.mxo + 32'(mx_i) + 32'(k))
        $fatal(1, "otpu_mxu: RMAX lane %0d address %0h, mxo %0h, mx_i %0d", k, mxl[k], h.mxo, mx_i);
`endif

  // read-modify-write pipeline for ACC: read now, data next cycle, (old*alpha)+new, write
  typedef struct packed {
    logic                   v;
    logic [NL-1:0]          m;
    logic [NL-1:0][31:0]    ad, nv;
    logic [NL-1:0][7:0]     col;
  } rmw_t;
  rmw_t r0, rw;                               // r0: data arriving now; rw: at the write stage
  rmw_t rx;                                   // RMAX (no ACC): drained lanes, compared next cycle
  f32_t ry [NL];
  // r1: r0 one granted cycle later, with the old values registered (xo): no path from the TMEM
  // block RAMs into the fmadd's DSP inputs in one cycle. The ASCALE factor (alr) is selected from
  // r0 as it moves into r1, so the fmadd's b operand is a flop too (no mux in front of the DSP);
  // c_asc and alpha cannot change while a valid entry sits in r0/r1 (the head changes only once
  // drained, alpha loads only before an ASCALE command drains)
  rmw_t r1;
  f32_t xo [NL], alr [NL];
  always_ff @(posedge clk)
    if (rst) r1 <= '0;
    else if (t_gnt) begin
      r1 <= r0;
      for (int k = 0; k < NL; k++) begin
        xo[k]  <= t_rdata[k];
        alr[k] <= c_asc ? alpha[r0.col[k][MW-2:0]] : F_ONE;
      end
    end
  for (genvar k = 0; k < NL; k++) begin : g_rmw
    otpu_fmadd #(.LM(LM), .LA(LA)) u_y (.clk, .en(t_gnt), .a(xo[k]), .b(alr[k]), .c(r1.nv[k]),
                                        .y(ry[k]));
  end
  otpu_delay #(.W($bits(rmw_t)), .N(LM + LA)) u_rw (.clk, .en(t_gnt), .d(r1), .q(rw));
  logic [3:0] rmw_n;                          // rows' lanes in flight (any nonzero = busy)

  // RMAX: the running max is kept as its sort key (mk = fkey(max); the max is always stored
  // ftz'd, so fkey(max) is the key it is compared by); written out through unkey.
  logic [31:0] mk [MCOLS];
  logic [MCOLS-1:0] mx_have;
  logic mx_done;
  function automatic f32_t unkey(input logic [31:0] k);
    return k[31] ? {1'b0, k[30:0]} : ~k;
  endfunction
  // rw's column hits, one-hot, computed from r1 and carried alongside u_rw (same length and
  // enable); the last stage is a reset flop, not an SRL tap. c_rmax is fixed while entries are
  // in flight (the head changes only once drained)
  logic [NL-1:0][MCOLS-1:0] rh, rh_p, rwh, rxh;
  always_comb
    for (int k = 0; k < NL; k++)
      for (int j = 0; j < MCOLS; j++) begin
        rh[k][j]  = r1.v && c_rmax && r1.m[k] && (r1.col[k][MW-2:0] == (MW-1)'(j));
        rxh[k][j] = rx.v && rx.m[k] && (rx.col[k][MW-2:0] == (MW-1)'(j));
      end
  otpu_delay #(.W(NL * MCOLS), .N(LM + LA - 1)) u_rwh (.clk, .en(t_gnt), .d(rh), .q(rh_p));
  always_ff @(posedge clk) if (rst) rwh <= '0; else if (t_gnt) rwh <= rh_p;
  // per column: the candidate key is selected by the hits alone, the compare only makes the
  // enable. At most one lane hits a column per cycle (lanes drain distinct columns) and rx / rw
  // are exclusive (rx only for !c_acc commands, rw only for c_acc ones)
  logic [MCOLS-1:0] mx_hit, mx_upd;
  logic [31:0] mx_cand [MCOLS];
  always_comb
    for (int j = 0; j < MCOLS; j++) begin
      mx_hit[j] = 1'b0; mx_upd[j] = 1'b0; mx_cand[j] = '0;
      for (int k = 0; k < NL; k++) begin
        if (rxh[k][j]) begin
          mx_hit[j] = 1'b1;
          mx_cand[j] = mx_cand[j] | fkey(rx.nv[k]);
          mx_upd[j] = mx_upd[j] | !mx_have[j] | (fkey(rx.nv[k]) > mk[j]);
        end
        if (rwh[k][j]) begin
          mx_hit[j] = 1'b1;
          mx_cand[j] = mx_cand[j] | fkey(ry[k]);
          mx_upd[j] = mx_upd[j] | !mx_have[j] | (fkey(ry[k]) > mk[j]);
        end
      end
    end
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && t_gnt)
      for (int j = 0; j < MCOLS; j++) begin
        int n;
        n = 0;
        for (int k = 0; k < NL; k++) n = n + int'(rxh[k][j]) + int'(rwh[k][j]);
        if (n > 1) $fatal(1, "otpu_mxu: several RMAX updates of column %0d in one cycle", j);
      end
`endif

  wire c_drained = c_act && !cl_ld && (pn || c_left == 0) && (rows_live == 0) && (rmw_n == 0) &&
                   !r0.v && !r1.v && !rx.v;
  // RMAX writes start a cycle after the head has drained: mx_q registers the test, so the
  // drained compares (c_left == 0, rows_live, rmw_n, ...) are off the TMEM write request and the
  // grant it feeds (clk125 at 125.49 MHz: c_left -> drained test -> MXU write request -> TMEM
  // arbitration -> VPU WBUF enable / alpha, mk enables, 14 levels, -0.116 ns). The test holds
  // while drained (nothing new enters the head until it completes, which needs mx_done), and it
  // is false in the completing cycle (mx_done), so a new head never sees a stale mx_q.
  logic mx_q;
  wire mx_go     = mx_q && !mx_done;
  wire c_fin     = c_drained && (!c_rmax || mx_done || c_tz);
  // the head completes (OVL: while the pop head may pop the next command's chunks)
  wire c_done    = c_fin && (OVL || !pop);
  wire al_go     = c_act && c_asc && al_st == 2'd0;

  always_comb begin
    t_ren = '0; t_raddr = '0; t_wen = '0; t_waddr = '0; t_wdata = '0;
    if (drain_go) begin
      for (int k = 0; k < LANES; k++) begin
        if (c_acc) begin
          t_ren[k] = (k < NL) && dln[k % NL];
          t_raddr[k] = daddr_l[k];
        end else begin
          t_wen[k] = (k < NL) && dln[k % NL];
          t_waddr[k] = daddr_l[k];
          t_wdata[k] = dval_l[k];
        end
      end
    end
    if (rw.v) begin
      for (int k = 0; k < NL; k++) begin
        if (rw.m[k]) begin
          t_wen[k] = 1'b1;
          t_waddr[k] = rw.ad[k];
          t_wdata[k] = ry[k];
        end
      end
    end
    // ASCALE factors and RMAX values move LANES words per cycle (consecutive words: distinct
    // banks)
    if (al_go) begin
      for (int k = 0; k < LANES; k++) begin
        if (32'(al_i) + 32'(k) < 32'(c_M)) begin
          t_ren[k] = 1'b1;
          t_raddr[k] = c_asa + 32'(al_i) + 32'(k);
        end
      end
    end
    if (mx_go) begin
      for (int k = 0; k < LANES; k++) begin
        if (mxm[k]) begin
          t_wen[k] = 1'b1;
          t_waddr[k] = mxl[k];
          t_wdata[k] = unkey(mk[MW'(32'(mx_i) + 32'(k))]);
        end
      end
    end
  end

  // ---- statistics for the profiler (per completed command): cycles the stream was starved
  // (work, no chunk), backpressured (chunks, no consumption), the drain frozen by the TMEM grant,
  // the issuer denied a DRAM port. The conditions are registered (st_c) and summed a cycle
  // late, off the grant paths: a command's count is st + g, g the last cycle's pending bit (none
  // after a command's end: that cycle's conditions belong to no command).
  logic [31:0] st_starve, st_bp, st_frz, st_deny;
  logic [3:0]  st_c, st_g;                  // {deny, frz, bp, starve}
  logic        st_f;                        // the last cycle ended a command
  assign st_g = st_f ? 4'd0 : st_c;

  always_ff @(posedge clk) begin
    done <= 1'b0;
    pf_u <= 1'b0;
    if (rst) begin
      i_act <= 1'b0;
      q_n <= '0; pn <= 1'b0; rows_p <= '0;
      occ <= '0;
      f_head <= '0; f_tail <= '0; f_count <= '0; f_rd <= '0; f_tl <= '0; f_rdl <= '0;
      s_head <= '0; s_tail <= '0; s_count <= '0; s_rd <= '0;
      ck <= '0; cg <= '0; dg <= '0; dg1 <= 8'd1; c_left <= '0; cl_ld <= 1'b0; rows_live <= '0;
      rf_q <= '0; rf_hs <= 1'b0; rf_t <= '0; rf_n <= '0; rf_nz <= 1'b0; rl_nz <= 1'b0;
      dj <= '0; mx_done <= 1'b0; mx_have <= '0;
      d_last <= 1'b0; d_fin <= 1'b0; ncnt <= '0; dln <= '0; mxm <= '0;
      al_st <= 2'd0; al_i <= '0; mx_i <= '0;
      r0 <= '0; rx <= '0; rmw_n <= '0;
      st_starve <= '0; st_bp <= '0; st_frz <= '0; st_deny <= '0;
      st_c <= '0; st_f <= 1'b0; mx_q <= 1'b0;
    end else begin
      logic [1:0] qn;
      logic [RFW:0] rn;
      logic [RFW:0] rl, rp;
      qent_t hn, nn;                            // the entries' next values (before a swap)
      qent_t hp, np;                            // the same without this cycle's accept (pop_q's:
                                                // a command accepted this cycle is not popped
                                                // before cl_ld, which reloads pop_q)
      logic pnx;
      qn = q_n;
      rn = rf_n;
      rl = rows_live;
      rp = rows_p;
      hn = h;
      nn = n;
      hp = h;
      np = n;
      pnx = pn;
      cl_ld <= 1'b0;
      if (cl_ld) c_left <= h.total;
      // ---- accept a command (the head if the queue is empty, else the next entry)
      if (start) begin
        if (q_n == 0) begin
          hn = cmd_e;                             // becomes the head now
          cl_ld <= 1'b1;
        end else nn = cmd_e;
        qn = qn + 1;
        if (cmd.w4[15:0] != 0 && cmd.w4[31:16] != 0) begin
          i_act  <= 1'b1;
          i_left <= cmd_total;
          i_KB   <= cmd_KBa;
          i_k    <= '0;
          i_rs   <= cmd.w5;
          i_srs  <= cmd.w7;
          i_unit <= cmd.flags[0];
          i_w4   <= (cmd.flags[5:4] != WF_W8);
          i_pair <= cmd_pair;
          row_addr <= cmd.w1; chunk_addr <= cmd.w1;
          srow_addr <= cmd.w2; scale_addr <= cmd.w2;
        end
      end
      // ---- release (in order): the head if not yet released, else the second entry
      if (go) begin
        if (q_n != 0 && !h.go) begin
          hn.go = 1'b1; hp.go = 1'b1;
        end else begin
          nn.go = 1'b1; np.go = 1'b1;
        end
      end
      // ---- issue one chunk request
      if (go_iss) begin
        i_left <= i_left - 1;
        if (i_left == 1) i_act <= 1'b0;
        if (i_k + 1 == i_KB) begin
          i_k <= '0;
          row_addr <= row_addr + i_rs;
          chunk_addr <= row_addr + i_rs;
          srow_addr <= srow_addr + i_srs;
          scale_addr <= srow_addr + i_srs;
        end else begin
          i_k <= i_k + 1;
          // 4-bit: two blocks per chunk (PAIR: both in one advance, with their two scale words)
          if (!i_w4 || i_pair || i_k[0]) chunk_addr <= chunk_addr + D;
          scale_addr <= scale_addr + (i_pair ? 32'd8 : 32'd4);
        end
      end
      // ---- FIFO pushes
      if (b_rvalid) begin
        f_tail <= f_tail + 1;
        for (int g = 0; g < FG; g++) f_tl[g] <= f_tl[g] + 1;
      end
      if (a_rvalid) begin
        s_tail <= s_tail + 1;
      end
      f_count <= f_count + (b_rvalid ? 1 : 0) - (fpop ? 1 : 0);
      occ <= occ + (b_req ? 1'b1 : 1'b0) - (fpop ? 1'b1 : 1'b0);
      s_count <= s_count + (a_rvalid ? 1 : 0) - ((pop && last_g && !c_unit) ? 1 : 0);
      // ---- pop one advance (the row's last group frees its chunk and scale)
      if (fpop) f_head <= f_head + 1;
      if (pop) begin
        if (last_g) begin
          if (!c_unit) s_head <= s_head + 1;
          c_left <= c_left - 1;
        end
        // the next read: the next entry, or back to the head for the row's next group
        if (last_k && !last_g) begin
          f_rd <= f_head;
          f_rdl <= {FG{f_head}};
        end else if (cdone) begin
          f_rd <= f_rd + 1;
          for (int g = 0; g < FG; g++) f_rdl[g] <= f_rdl[g] + 1;
        end
        if (!c_unit) s_rd <= (last_k && !last_g) ? s_head : s_rd + 1;
        if (ck == 0) begin
          if (!OVL || !pn) rl = rl + 1;
          else rp = rp + 1;
        end
        ck <= last_k ? '0 : ck + 1;
        if (last_k) cg <= last_g ? '0 : cg + 1;
      end
      // ---- a finished row enters the result FIFO
      if (rf_push) begin
        rf_t <= rf_t + 1;
        rn = rn + 1;
      end
      // ---- drain (holds while the TMEM grant is withheld)
      if (t_gnt) begin
        r0 <= '0;
        rx <= '0;
        if (drain_go) begin
          if (c_acc) begin
            r0.v <= 1'b1;
            r0.m <= dln;
            r0.ad <= daddr_l[NL-1:0];
            r0.nv <= dval_l[NL-1:0];
            r0.col <= dcol_l[NL-1:0];
          end
          if (c_rmax && !c_acc) begin
            rx.v <= 1'b1;
            rx.m <= dln;
            rx.nv <= dval_l[NL-1:0];
            rx.col <= dcol_l[NL-1:0];
          end
          if (drain_row_done) begin
            dj <= '0;
            if (d_last) begin                       // the next weight row, group 0
              dg <= '0; dg1 <= 8'd1;
              for (int j = 0; j < MCOLS; j++) begin
                logic [31:0] a;
                a = drow[j] + 32'd1;
                dad[j] <= a;
                drow[j] <= a;
                if (j < NL) dal[j] <= a;
              end
            end else begin                          // the next group of this row
              dg <= dg + 8'd1; dg1 <= dg1 + 8'd1;
              for (int j = 0; j < MCOLS; j++) begin
                logic [31:0] a;
                a = dad[j] + h.gs;
                dad[j] <= a;
                if (j < NL) dal[j] <= a;
              end
            end
            rn = rn - 1;
            rl = rl - 1;
          end else begin
            dj <= dj + ncnt;
            dal <= dlane_a(dad, dj + ncnt);
          end
        end
        // RMAX (rx: registered so the lane selection and the max compare are in different cycles)
        for (int j = 0; j < MCOLS; j++) begin
          if (mx_upd[j]) mk[j] <= mx_cand[j];
          if (mx_hit[j]) mx_have[j] <= 1'b1;
        end
        rmw_n <= rmw_n + ((drain_go && c_acc) ? 4'd1 : 4'd0) - (rw.v ? 4'd1 : 4'd0);
        if (mx_go) begin
          if (32'(mx_i) + LANES >= 32'(c_M)) mx_done <= 1'b1;
          else mx_i <= mx_i + 8'(LANES);
        end
        if (al_go) al_st <= 2'd1;
        if (al_st == 2'd1) begin
          // one write per entry (a constant index): a variable-index write of several lanes makes
          // Vivado try, and crash while dissolving, a RAM for these MCOLS registers
          for (int j = 0; j < MCOLS; j++)
            if (32'(j) >= 32'(al_i) && 32'(j) < 32'(al_i) + LANES)
              alpha[j] <= t_rdata[$clog2(LANES)'(32'(j) - 32'(al_i))];
          if (32'(al_i) + LANES >= 32'(c_M)) al_st <= 2'd2;
          else begin
            al_i <= al_i + 8'(LANES);
            al_st <= 2'd0;
          end
        end
      end
      rf_n <= rn;
      rf_q <= rf_q + RFW'(rf_hs);
      rf_hs <= d_pop;
      // RMAX lane addresses: they advance with mx_i while the writes run, else h.mxo + k
      for (int k = 0; k < LANES; k++)
        if (!mx_go) mxl[k] <= h.mxo + 32'(k);
        else if (t_gnt && 32'(mx_i) + LANES < 32'(c_M)) mxl[k] <= mxl[k] + 32'(LANES);
      rows_live <= rl;
      rows_p <= rp;
      // rn != 0, rl != 0 from the counts' own zero / one tests: the grant enters last
      rf_nz <= rf_push || (rf_nz && !(rf_n == 1 && d_pop));
      rl_nz <= (pop && ck == 0 && (!OVL || !pn)) || (rl_nz && !(rows_live == 1 && d_pop));
      // OVL: the pop head moves to the next command once the head's chunks have all popped, if
      // the pipeline treats both alike (PAIR, and M under PAIR: pr0, hi0 and M0 are read late)
      if (OVL && !pn && !cl_ld && c_left == 0 && q_n == 2'd2 && n.go && !c_done &&
          n.pair == h.pair && (!h.pair || n.M == h.M)) begin
        pnx = 1'b1;
        c_left <= n.total;
      end
      // ---- statistics
      st_c <= {want_iss && !go_iss, !t_gnt && (drain_go || rw.v || mx_go || al_go),
               more && f_count != 0 && !pop, more && f_count == 0};
      st_f <= c_done;
      mx_q <= c_drained && c_rmax && !mx_done && !c_tz;
      st_starve <= st_starve + 32'(st_g[0]);
      st_bp <= st_bp + 32'(st_g[1]);
      st_frz <= st_frz + 32'(st_g[2]);
      st_deny <= st_deny + 32'(st_g[3]);
      // ---- the consumer's command is complete
      if (c_done) begin
        done <= 1'b1;
        qn = qn - 1;
        dg <= '0; dg1 <= 8'd1;
        if (!pn) begin                            // the pop head moves along
          ck <= '0; cg <= '0;
          // the next head's chunk count: the queued entry's, or (cl_ld) the total of a command
          // accepted this cycle, a cycle later
          c_left <= (q_n == 2'd2) ? n.total : '0;
          if (start && q_n == 2'd1) cl_ld <= 1'b1;
        end else begin                            // it already runs the next command: its rows
          rows_live <= rp;
          rl_nz <= (rp != 0);
          rows_p <= '0;
        end
        pnx = 1'b0;                               // the next command is both heads now
        dj <= '0;
        for (int j = 0; j < MCOLS; j++) begin
          logic [31:0] a;
          a = (start && q_n == 2'd1) ? cmd.w3 + 32'(j) * 32'(cmd.w6[15:0]) : n.out + n.jo[j];
          dad[j] <= a;
          drow[j] <= a;
          if (j < NL) dal[j] <= a;
        end
        mx_done <= 1'b0; mx_have <= '0;
        al_st <= 2'd0; al_i <= '0; mx_i <= '0;
        pf_u <= 1'b1;
        pf_uv <= {st_deny + 32'(st_g[3]), st_frz + 32'(st_g[2]), st_bp + 32'(st_g[1]),
                  st_starve + 32'(st_g[0])};
        st_starve <= '0; st_bp <= '0; st_frz <= '0; st_deny <= '0;
      end
      // the head's output base (set when a command becomes head)
      if (start && q_n == 0) begin
        for (int j = 0; j < MCOLS; j++) begin
          logic [31:0] a;
          a = cmd.w3 + 32'(j) * 32'(cmd.w6[15:0]);
          dad[j] <= a;
          drow[j] <= a;
          if (j < NL) dal[j] <= a;
        end
      end
      // the entries: completing the head swaps them (the next entry, or the command accepted
      // this cycle, becomes the head; the old head stays in n)
      begin
        qent_t hx, nx, hpx, npx;
        if (c_done) begin
          hx = nn; nx = hn; hpx = np; npx = hp;
        end else begin
          hx = hn; nx = nn; hpx = hp; npx = np;
        end
        h <= hx; n <= nx;
        // the drain's next step: a new head starts at its first row's group 0 (all of it if the
        // row is one group); a drain step moves on in the group, or to the next group / row
        if (c_done || (start && q_n == 0)) begin
          logic [MW:0] st;
          st = hx.g1 ? {hx.dfl, hx.dnl} : dstep(MW'(MCOLS), hx.run);
          d_last <= hx.g1;
          {d_fin, ncnt} <= st;
          dln <= dlanes(st[MW-1:0]);
          for (int k = 0; k < LANES; k++) mxm[k] <= 32'(k) < 32'(hx.M);
        end else begin
          if (t_gnt && drain_go) begin
            if (!d_fin) begin
              {d_fin, ncnt} <= st_mid;
              dln <= ln_mid;
            end else begin
              d_last <= d_nl;
              {d_fin, ncnt} <= st_grp;
              dln <= ln_grp;
            end
          end
          if (t_gnt && mx_go && 32'(mx_i) + LANES < 32'(c_M)) mxm <= mxm_nx;
        end
        // the pop head's entry (OVL), from the entries, not from the command being accepted: a
        // command that becomes the head as it is accepted gets its entry a cycle later (cl_ld;
        // the pipeline is empty then, and nothing pops before c_left loads)
        pop_q <= pnx ? npx : hpx;
      end
      pn <= pnx;
      q_n <= qn;
    end
  end
`ifndef SYNTHESIS
  // the head / next mapping relies on the rdy contract (start only with a free entry), and a
  // command never completes while it pops (c_fin needs c_left == 0, pop c_left != 0)
  always @(posedge clk) if (!rst) begin
    if (start && q_n >= 2'd2) $fatal(1, "otpu_mxu: start with %0d commands queued", q_n);
    if (!OVL && c_fin && pop) $fatal(1, "otpu_mxu: the head completes while it pops");
    if (OVL && q_n != 0 && !cl_ld && pop_q != (pn ? n : h))
      $fatal(1, "otpu_mxu: pop_q is not the pop head's entry");
  end
`endif

endmodule

// Simple dual-port block RAM, registered read with enable (read-first: a read of the word
// written at the same edge returns the old word).
module otpu_ram_sdp #(parameter int W = 32, parameter int N = 1024) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  (* ram_style = "block" *) logic [W-1:0] mem [N];
  always_ff @(posedge clk) if (we) mem[wa] <= wd;
  always_ff @(posedge clk) if (re) rd <= mem[ra];
endmodule

// N-cycle delay line with an enable: no reset and no forced last flip-flop (unlike otpu_delay), so
// a long one maps to shift-register LUTs and its last stage may move into a DSP48's input register
// (the systolic MXU's operand skews)
module otpu_skew #(parameter int W = 8, parameter int N = 1) (
  input  logic         clk,
  input  logic         en,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  if (N == 0) begin : g_wire
    assign q = d;
  end else begin : g_regs
    logic [W-1:0] r [N];
    always_ff @(posedge clk) if (en) begin
      r[0] <= d;
      for (int k = 1; k < N; k++) r[k] <= r[k-1];
    end
    assign q = r[N-1];
  end
endmodule

// One position of the systolic MXU (IMPL 2): a DSP48E1 with the activation in its B register
// (BREG 1), the two weight streams on A (low) and D (high), one selected by the pre-adder's input
// gates (INMODE: sel 0 -> A, 1 -> D), the product in M and the chain's running sum in P:
// P = PCIN + M (FIRST: P = M). A chain's LAST position leaves P unregistered: the column end's A
// / D input registers take its place (otpu_colend), so the route to the column end is not in
// front of its pre-adder and multiplier. The select is registered in the DSP (INMODEREG, loaded
// with en): sel is its next value, so the fabric's route to the DSP column ends at the INMODE
// register instead of before the pre-adder and multiplier (133.33 MHz, 812bb01: hi0 -> INMODE,
// 0 levels, 4.4 ns of route and a 2.4 ns setup, +0.174 ns). WREG registers the weights too (A
// and D registers, loaded with en: wlo and whi are their next values). Simulation uses the
// equivalent behavioural model.
module otpu_pe #(parameter bit FIRST = 1'b0, parameter bit LAST = 1'b0, parameter bit WREG = 1'b0) (
  input  logic               clk,
  input  logic               en,
  input  logic [7:0]         act,
  input  logic [7:0]         wlo,               // WREG: the weights from the next cycle on
  input  logic [7:0]         whi,
  input  logic               sel,               // the select from the next cycle on
  input  logic [47:0]        pcin,
  output logic [47:0]        pcout,
  output logic signed [23:0] p
);
`ifdef SYNTHESIS
  logic [47:0] pf;
  DSP48E1 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .USE_DPORT("TRUE"), .USE_MULT("MULTIPLY"),
    .USE_SIMD("ONE48"), .AREG(WREG ? 1 : 0), .ACASCREG(WREG ? 1 : 0), .BREG(1), .BCASCREG(1),
    .CREG(0), .DREG(WREG ? 1 : 0),
    .ADREG(0), .MREG(1), .PREG(LAST ? 0 : 1), .INMODEREG(1), .OPMODEREG(0), .ALUMODEREG(0),
    .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_dsp (
    .CLK(clk),
    .A({{22{wlo[7]}}, wlo}), .B({{10{act[7]}}, act}), .C(48'd0), .D({{17{whi[7]}}, whi}),
    .INMODE({2'b00, sel, sel, 1'b0}), .OPMODE(FIRST ? 7'b000_01_01 : 7'b001_01_01),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b0), .CEA2(WREG ? en : 1'b0), .CEB1(en), .CEB2(en), .CEC(1'b0),
    .CED(WREG ? en : 1'b0), .CEAD(1'b0),
    .CEM(en), .CEP(LAST ? 1'b0 : en), .CEALUMODE(1'b0), .CECTRL(1'b0), .CECARRYIN(1'b0),
    .CEINMODE(en),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(30'd0), .BCIN(18'd0), .PCIN(pcin), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(pcout), .P(pf), .CARRYCASCOUT(), .MULTSIGNOUT(), .CARRYOUT(),
    .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT());
  assign p = pf[23:0];
`else
  logic [7:0] br, wlr, whr;
  logic       selr;
  logic signed [15:0] m;
  wire  signed [23:0] pn = (FIRST ? 24'sd0 : $signed(pcin[23:0])) + 24'(m);
  wire  [7:0] wl = WREG ? wlr : wlo, wh = WREG ? whr : whi;
  always_ff @(posedge clk) if (en) begin
    br <= act;
    selr <= sel;
    wlr <= wlo;
    whr <= whi;
    m <= 16'(int'($signed(br)) * int'($signed(selr ? wh : wl)));
  end
  if (LAST) begin : g_comb
    assign p = pn;
  end else begin : g_preg
    always_ff @(posedge clk) if (en) p <= pn;
  end
  assign pcout = LAST ? 48'd0 : 48'(p);              // (a chain's last cascades nowhere)
`endif
endmodule

// The systolic MXU's column end (IMPL 2): four DSP48E1s, one per sub-block b. DSP b registers the
// sub-block's chain ends (a[b], d[b]: the chains' last P, which otpu_pe LAST leaves unregistered)
// in its A and D registers, sums them in its pre-adder, multiplies by m_b (its B register:
// mb[4b+3:4b], unsigned, loaded with a / d) into M (v[b]); the block sum runs down the cascade in
// the same cycle, P = PCIN + M (DSP 0: P = M), and DSP 3 registers it (PREG):
//   s = v0 + v1 + v2 + v3   (one cycle after v, as a fabric or inferred sum would)
// The cascade keeps the sum off the fabric: Vivado mapped the inferred sum into three DSP ALUs
// joined through the C port (two fabric routes, 6.76 ns: -0.479 ns at 133.33 MHz). The input
// registers keep the route from the chain ends (3.5 ns in the OOC) out of the multiplier's cycle:
// the chain ends' P -> v was 0.008 ns from failing out of context at 133.33 MHz. Exact: every
// value fits 48 bits, s is the sum's low SW bits. Simulation uses the equivalent behavioural model.
module otpu_colend #(parameter int SW = 23) (
  input  logic                 clk,
  input  logic                 en,
  input  logic [3:0][23:0]     a,        // sub-block b's chain ends: the pre-adder's A ...
  input  logic [3:0][23:0]     d,        // ... and D (signed)
  input  logic [15:0]          mb,       // m_3 .. m_0, with a / d
  output logic signed [SW-1:0] s
);
  // the pre-adder is 25 bits wide: exact for sums of SW <= 25 bits
  initial if (SW > 25) $fatal(1, "otpu_colend: SW %0d > 25", SW);
`ifdef SYNTHESIS
  logic [3:0][47:0] pc, pf;
  for (genvar b = 0; b < 4; b++) begin : g_d
    DSP48E1 #(
      .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .USE_DPORT("TRUE"), .USE_MULT("MULTIPLY"),
      .USE_SIMD("ONE48"), .AREG(1), .ACASCREG(1), .BREG(1), .BCASCREG(1), .CREG(0), .DREG(1),
      .ADREG(0), .MREG(1), .PREG(b == 3 ? 1 : 0), .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
      .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
      .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
      .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
    ) u_dsp (
      .CLK(clk),
      .A({{6{a[b][23]}}, a[b]}), .B({14'd0, mb[4*b +: 4]}), .C(48'd0), .D({d[b][23], d[b]}),
      .INMODE(5'b00100), .OPMODE(b == 0 ? 7'b000_01_01 : 7'b001_01_01),
      .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
      .CEA1(en), .CEA2(en), .CEB1(en), .CEB2(en), .CEC(1'b0), .CED(en), .CEAD(1'b0),
      .CEM(en), .CEP(b == 3 ? en : 1'b0), .CEALUMODE(1'b0), .CECTRL(1'b0), .CECARRYIN(1'b0),
      .CEINMODE(1'b0),
      .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
      .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
      .ACIN(30'd0), .BCIN(18'd0), .PCIN(b == 0 ? 48'd0 : pc[b == 0 ? 0 : b - 1]),
      .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
      .ACOUT(), .BCOUT(), .PCOUT(pc[b]), .P(pf[b]), .CARRYCASCOUT(), .MULTSIGNOUT(),
      .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT());
  end
  assign s = $signed(pf[3][SW-1:0]);
`else
  logic [3:0][23:0] ar, dr;
  logic [15:0] mr;
  logic signed [47:0] v [4];
  always_ff @(posedge clk) if (en) begin
    ar <= a; dr <= d; mr <= mb;
    for (int b = 0; b < 4; b++)
      v[b] <= (48'($signed(ar[b])) + 48'($signed(dr[b]))) * 48'(mr[4*b +: 4]);
    s <= SW'(v[0] + v[1] + v[2] + v[3]);
  end
`endif
endmodule
