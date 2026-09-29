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
  logic [31:0] q_out [2], q_total [2];
  logic        q_tz [2];                      // q_total == 0 (registered: off the drain path)
  logic [15:0] q_KB [2], q_KBa [2], q_ors [2];   // KBa: advances per row
  logic        q_pair [2];
  logic [MCOLS-1:0] q_hi [2];                 // PAIR: the columns that take the odd blocks (j >= M)
  logic [31:0] q_mxo [2];                     // RMAX output base: out + M * ors (no multiply later)
  logic [7:0]  q_M [2], q_ab [2];
  logic [MW-1:0] q_run [2];                  // drain lanes per cycle without a bank conflict
  logic        q_unit [2], q_acc [2], q_rmax [2], q_asc [2], q_go [2];
  logic [1:0]  q_wf [2];
  logic [31:0] q_asa [2];
  logic [31:0] q_jo [2][MCOLS];            // j * ors
  logic [7:0]  q_G [2];                     // groups: ceil(M / MCOLS)
  logic [MW-1:0] q_Ml [2];                  // rows of the last group
  logic [31:0] q_gs [2];                    // drain address step to the next group
                                            // (MCOLS * ors)
  // Two heads (IMPL 2: commands overlap in the pipeline; docs/mxu_systolic.md). q_h is the drain's
  // head, the oldest command: its results are drained, its completion signalled. q_p is the pop
  // head: q_h, or, once the head's chunks have all popped, the next command if the pipeline treats
  // it alike (the same PAIR and M, the only per-command state read after S0). The next command's
  // rows then enter the pipeline behind the head's instead of after its drain. OVL = 0 keeps
  // q_p = q_h (one command in the pipeline at a time).
  localparam bit OVL = (IMPL == 2);
  logic        q_h, q_pr;
  wire         q_p = OVL ? q_pr : q_h;
  logic [1:0]  q_n;

  wire [31:0] c_out = q_out[q_h];
  wire        c_tz = q_tz[q_h];
  // the pop head's (c_KB .. c_wf, p_*) and the drain head's (the rest)
  wire [15:0] c_KB = q_KB[q_p], c_KBa = q_KBa[q_p];
  wire        c_pair = q_pair[q_p];
  wire [MCOLS-1:0] c_hi = q_hi[q_p];
  wire [7:0]  c_M = q_M[q_h], c_ab = q_ab[q_p];
  wire [7:0]  p_M = q_M[q_p], p_G = q_G[q_p];
  wire        p_act = (q_n != 0) && q_go[q_p];
  wire [MW-1:0] c_run = q_run[q_h];
  wire        c_unit = q_unit[q_p], c_acc = q_acc[q_h], c_rmax = q_rmax[q_h];
  wire        c_asc = q_asc[q_h];
  wire [1:0]  c_wf = q_wf[q_p];
  wire        c_w4 = (c_wf != WF_W8);
  wire [31:0] c_asa = q_asa[q_h];
  wire        c_act = (q_n != 0) && q_go[q_h];
  wire [7:0]  c_G = q_G[q_h];
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

  // ================================================================== consumer control
  logic [15:0] ck;
  logic [7:0]  cg;                          // the group of the row being consumed (replay)
  logic [31:0] c_left;                      // chunks of the pop head not yet freed
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
  assign b_addr = b_req ? (chunk_addr >> 2) : '0;
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
  (* keep_hierarchy = "yes" *)
  otpu_ram_sdp #(.W(D * 8), .N(DEPTH)) u_fd (
    .clk, .we(b_rvalid), .wa(f_tail), .wd(b_rdata), .re(en_c), .ra(f_rd), .rd(w0));

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
    // delayed by its chain stage k = i % CL, and the weights then move one register hop per
    // column: column j sees them j cycles after column 0 (fan-out 2, no broadcast). Both streams
    // flow: the low block (or the 4-bit half of a non-PAIR advance) and PAIR's high block; each
    // column takes one by its per-command hi0[j]. Column j's activation byte i is delayed
    // 1 + k + j (one shift register per bit). Along D: chains of CL products, one per DSP48E1
    // (M register, P = PCIN + M). The 4-bit sub-blocks are two chains each, so the sub-block
    // multipliers apply at the chain ends, as in IMPL 0; column j's s4 is then delayed
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
    for (genvar j = 0; j < MCOLS; j++) begin : g_ask
      for (genvar i = 0; i < D; i++) begin : g_p
        otpu_skew #(.W(8), .N(1 + i % CLS + j)) u_a (.clk, .en(en_c), .d(a0[(j*D + i)*8 +: 8]),
                                                   .q(as_k[j][i]));
      end
    end
    logic signed [15:0] mreg [MCOLS][D];                // products (M registers)
    logic signed [23:0] preg [MCOLS][D];                // running sums (P registers)
    always_ff @(posedge clk) if (en_c)
      for (int j = 0; j < MCOLS; j++)
        for (int i = 0; i < D; i++) begin
          mreg[j][i] <= 16'(int'($signed(as_k[j][i])) *
                            int'($signed(hi0[j] ? wc[j][i][15:8] : wc[j][i][7:0])));
          preg[j][i] <= ((i % CLS == 0) ? 24'sd0 : preg[j][i - 1]) + 24'(mreg[j][i]);
        end
    // chain ends (S0 + CLS + 2 + j): the sub-block sums times their multipliers, then the block
    // sum; the multipliers of column j (its block's under PAIR) travel alongside
    logic [SW-1:0] vs [MCOLS][4];
    logic signed [SW-1:0] s4r [MCOLS];
    for (genvar j = 0; j < MCOLS; j++) begin : g_col_end
      logic [15:0] mbd;
      otpu_skew #(.W(16), .N(CLS + 2 + j)) u_mb (.clk, .en(en_c), .d(hi0[j] ? mb0h : mb0),
                                               .q(mbd));
      always_ff @(posedge clk) if (en_c) begin
        for (int b = 0; b < 4; b++) begin
          logic [SW-1:0] t;
          t = '0;
          for (int c = 0; c < CPS; c++) t = t + SW'(preg[j][(b*CPS + c)*CLS + CLS - 1]);
          vs[j][b] <= SW'(t * SW'(mbd[4*b +: 4]));
        end
        s4r[j] <= $signed(vs[j][0] + vs[j][1] + vs[j][2] + vs[j][3]);
      end
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
  f32_t        rf_v [RF][MCOLS];
  logic [RFW-1:0] rf_h, rf_t;
  logic [RFW:0]   rf_n;
  wire         rf_push = en_c && lv2;

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
  wire           d_last = (dg1 == c_G);
  wire  [MW-1:0] c_Mn = d_last ? q_Ml[q_h] : MW'(MCOLS);   // its results
  logic [LANES-1:0][31:0] daddr_l, dval_l;
  logic [LANES-1:0][7:0]  dcol_l;
  // (OVL: the result FIFO's head row is the head command's while it has rows not yet drained)
  wire  drain_go = (rf_n != 0) && (!OVL || rows_live != 0) && (!c_asc || al_st == 2'd2);
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && drain_go && dg1 != dg + 8'd1) $fatal(1, "otpu_mxu: dg1 %0d, dg %0d", dg1, dg);
`endif
  // Results j and j' of a row sit ors * (j' - j) words apart: the same bank iff that is a multiple
  // of LANES. So consecutive results are conflict-free in runs of LANES / gcd(ors, LANES)
  // (capped at NL), a per-command constant: the lane count is min(run, M - dj), with no serial
  // bank check between the result FIFO and the TMEM request.
  wire  [MW-1:0] d_left = c_Mn - dj;
  assign ncnt = (c_run < d_left) ? c_run : d_left;
  // (the lanes' addresses and values do not wait for the count: only the enables do)
  always_comb begin
    daddr_l = '0; dval_l = '0; dcol_l = '0;
    for (int k = 0; k < NL; k++)
      if (32'(dj) + 32'(k) < MCOLS) begin
        daddr_l[k] = dad[MW'(dj) + MW'(k)];
        dval_l[k] = rf_v[rf_h][MW'(dj) + MW'(k)];
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
  wire drain_row_done = drain_go && (MW'(dj + ncnt) == c_Mn);   // dj + ncnt <= M

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
  // c_asc and alpha cannot change while a valid entry sits in r0/r1 (q_h flips only once drained,
  // alpha loads only before an ASCALE command drains)
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
  // in flight (q_h flips only once drained)
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

  wire c_drained = c_act && (q_p != q_h || c_left == 0) && (rows_live == 0) && (rmw_n == 0) &&
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
          t_ren[k] = 32'(k) < 32'(ncnt);
          t_raddr[k] = daddr_l[k];
        end else begin
          t_wen[k] = 32'(k) < 32'(ncnt);
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
        if (32'(mx_i) + 32'(k) < 32'(c_M)) begin
          t_wen[k] = 1'b1;
          t_waddr[k] = q_mxo[q_h] + 32'(mx_i) + 32'(k);
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
      q_h <= 1'b0; q_pr <= 1'b0; q_n <= '0; rows_p <= '0;
      occ <= '0;
      f_head <= '0; f_tail <= '0; f_count <= '0; f_rd <= '0;
      s_head <= '0; s_tail <= '0; s_count <= '0; s_rd <= '0;
      ck <= '0; cg <= '0; dg <= '0; dg1 <= 8'd1; c_left <= '0; rows_live <= '0;
      rf_h <= '0; rf_t <= '0; rf_n <= '0;
      dj <= '0; mx_done <= 1'b0; mx_have <= '0;
      al_st <= 2'd0; al_i <= '0; mx_i <= '0;
      r0 <= '0; rx <= '0; rmw_n <= '0;
      st_starve <= '0; st_bp <= '0; st_frz <= '0; st_deny <= '0;
      st_c <= '0; st_f <= 1'b0; mx_q <= 1'b0;
    end else begin
      logic [1:0] qn;
      logic [RFW:0] rn;
      logic [RFW:0] rl, rp;
      qn = q_n;
      rn = rf_n;
      rl = rows_live;
      rp = rows_p;
      // ---- accept a command
      if (start) begin
        logic qi;
        qi = q_h ^ (q_n != 0);
        q_out[qi]   <= cmd.w3;
        q_total[qi] <= cmd_total;
        q_tz[qi]    <= cmd_tz;
        if (q_n == 0) c_left <= cmd_total;          // becomes the head now
        q_KB[qi]    <= cmd.w4[31:16];
        q_KBa[qi]   <= cmd_KBa;
        q_pair[qi]  <= cmd_pair;
        for (int j = 0; j < MCOLS; j++) q_hi[qi][j] <= cmd_pair && 32'(j) >= 32'(cmd.w6[23:16]);
        q_ors[qi]   <= cmd.w6[15:0];
        q_mxo[qi]   <= cmd.w3 + 32'(cmd.w6[23:16]) * 32'(cmd.w6[15:0]);
        q_M[qi]     <= cmd.w6[23:16];
        q_run[qi]   <= drain_run(cmd.w6[15:0]);
        q_ab[qi]    <= cmd.w6[31:24];
        q_unit[qi]  <= cmd.flags[0];
        q_wf[qi]    <= cmd.flags[5:4];
        q_acc[qi]   <= cmd.flags[1];
        q_rmax[qi]  <= cmd.flags[2];
        q_asc[qi]   <= cmd.flags[3];
        q_go[qi]    <= 1'b0;
        q_asa[qi]   <= cmd.w2;
        for (int j = 0; j < MCOLS; j++) q_jo[qi][j] <= 32'(j) * 32'(cmd.w6[15:0]);
        begin
          int g;
          g = (int'(cmd.w6[23:16]) + MCOLS - 1) / MCOLS;
          if (g < 1) g = 1;
          q_G[qi]  <= 8'(g);
          q_Ml[qi] <= MW'(int'(cmd.w6[23:16]) - (g - 1) * MCOLS);
          q_gs[qi] <= 32'(MCOLS) * 32'(cmd.w6[15:0]);
        end
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
        if (q_n != 0 && !q_go[q_h]) q_go[q_h] <= 1'b1;
        else q_go[~q_h] <= 1'b1;
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
      if (b_rvalid) f_tail <= f_tail + 1;
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
        if (last_k && !last_g) f_rd <= f_head;
        else if (cdone) f_rd <= f_rd + 1;
        if (!c_unit) s_rd <= (last_k && !last_g) ? s_head : s_rd + 1;
        if (ck == 0) begin
          if (!OVL || q_p == q_h) rl = rl + 1;
          else rp = rp + 1;
        end
        ck <= last_k ? '0 : ck + 1;
        if (last_k) cg <= last_g ? '0 : cg + 1;
      end
      // ---- a finished row enters the result FIFO
      if (rf_push) begin
        for (int j = 0; j < MCOLS; j++) rf_v[rf_t][j] <= rowr[j];
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
            for (int k = 0; k < NL; k++) r0.m[k] <= (32'(k) < 32'(ncnt));
            r0.ad <= daddr_l[NL-1:0];
            r0.nv <= dval_l[NL-1:0];
            r0.col <= dcol_l[NL-1:0];
          end
          if (c_rmax && !c_acc) begin
            rx.v <= 1'b1;
            for (int k = 0; k < NL; k++) rx.m[k] <= (32'(k) < 32'(ncnt));
            rx.nv <= dval_l[NL-1:0];
            rx.col <= dcol_l[NL-1:0];
          end
          if (drain_row_done) begin
            dj <= '0;
            if (d_last) begin                       // the next weight row, group 0
              dg <= '0; dg1 <= 8'd1;
              for (int j = 0; j < MCOLS; j++) begin
                dad[j] <= drow[j] + 32'd1;
                drow[j] <= drow[j] + 32'd1;
              end
            end else begin                          // the next group of this row
              dg <= dg + 8'd1; dg1 <= dg1 + 8'd1;
              for (int j = 0; j < MCOLS; j++) dad[j] <= dad[j] + q_gs[q_h];
            end
            rf_h <= rf_h + 1;
            rn = rn - 1;
            rl = rl - 1;
          end else begin
            dj <= dj + ncnt;
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
      rows_live <= rl;
      rows_p <= rp;
      // OVL: the pop head moves to the next command once the head's chunks have all popped, if
      // the pipeline treats both alike (PAIR, and M under PAIR: pr0, hi0 and M0 are read late)
      if (OVL && q_p == q_h && c_left == 0 && q_n == 2'd2 && q_go[~q_h] && !c_done &&
          q_pair[~q_h] == q_pair[q_h] && (!q_pair[q_h] || q_M[~q_h] == q_M[q_h])) begin
        q_pr <= ~q_h;
        c_left <= q_total[~q_h];
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
        q_h <= ~q_h;
        qn = qn - 1;
        dg <= '0; dg1 <= 8'd1;
        if (q_p == q_h) begin                     // the pop head moves along
          q_pr <= ~q_h;
          ck <= '0; cg <= '0;
          // the next head's chunk count: the queued entry, or a command accepted this cycle
          c_left <= (q_n == 2'd2) ? q_total[~q_h] : (start ? cmd_total : '0);
        end else begin                            // it already runs the next command: its rows
          rows_live <= rp;
          rows_p <= '0;
        end
        dj <= '0;
        for (int j = 0; j < MCOLS; j++) begin
          dad[j] <= (start && q_n == 2'd1) ? cmd.w3 + 32'(j) * 32'(cmd.w6[15:0])
                                           : q_out[~q_h] + q_jo[~q_h][j];
          drow[j] <= (start && q_n == 2'd1) ? cmd.w3 + 32'(j) * 32'(cmd.w6[15:0])
                                            : q_out[~q_h] + q_jo[~q_h][j];
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
          dad[j] <= cmd.w3 + 32'(j) * 32'(cmd.w6[15:0]);
          drow[j] <= cmd.w3 + 32'(j) * 32'(cmd.w6[15:0]);
        end
      end
      q_n <= qn;
    end
  end

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
