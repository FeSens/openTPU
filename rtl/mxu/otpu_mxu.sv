// MXU, the MX column (docs/isa.md MM, docs/mx_mxu.md): streams N rows x KB blocks of weights from
// DRAM (port B, one D-byte chunk per cycle, prefetched through a FIFO) and their scale rows (port
// A, 32-byte lines), and reuses each chunk across up to MCOLS stationary rows held in ACT RAM. Each
// output is one exact integer, accumulated in its column end over the whole row:
//   c   = sum_{i in g} act[j][i] * w[i]              group g: D/8 elements, a chain of the array
//   sh  = W - dw(n, k, g) - da[j][k][g/2]            W 16 (int8x), 11 (int4x, fp4x6), 13 (nvfp4)
//   acc = sum_k sum_g (sh >= 0 ? m(n, k, g) << sh : 0) * c                 48 bits, exact
// and one fp32 epilogue per output, in E lanes the columns share (the result FIFO between them):
//   y   = (i2f(acc) * Sw[n]) * (Sa[j] * 2^-W)          then ACC / ASCALE / RMAX as before
// 4-bit weights (flags[5:4] = WF: 1 int4, 2 E2M1, 3 NVFP4): a D-byte chunk holds two blocks (the
// low half first) and is consumed in two advances. PAIR (flags[6], 4-bit, 2M <= MCOLS; "Column
// reuse"): a chunk is consumed in one advance, columns j < M take its low block 2c (ACT block
// ab+2c), columns j+M its high block 2c+1 (ACT block ab+2c+1); the two columns accumulate
// separately and their integers are added before the result FIFO (+0 for the missing block of an
// odd KB: its multipliers are 0). RSC (flags[7]): each scale row is Sw alone. UNIT: Sw 1, dw 0, m 1.
//
// The array is the systolic one (docs/mxu_systolic.md): chains of CLS = min(CL, D/8) products
// (DSP48 cascades), so that every chain end is one group. A column end (otpu_mxce, rtl/tech) takes
// the column's eight group sums, multiplies each by its B = m << sh (the shift in fabric, two
// cycles ahead), sums them down a cascade and accumulates the row in a ninth DSP. The columns are
// skewed by a cycle each; their row sums are deskewed and registered, PAIR's halves added, and the
// row (MCOLS integers, the rows' Sa * 2^-W and Sw) written to the result FIFO. The drain picks up
// to E results of the head row a cycle (fewer at a TMEM bank conflict) into the lanes: i2f48 (round
// to nearest even), the two multiplies, then the TMEM write or the read-modify-write (ACC, ASCALE)
// as before; everything after the result FIFO advances only while the TMEM grant is given.
//
// The scale reader: for every advance the issuer asks port A for the 32-byte line that holds the
// advance's block data (int8x: one byte; int4x / fp4x6: a word, PAIR two; NVFP4: eight bytes,
// PAIR sixteen; RSC: Sw), and for a row's Sw: in the same line when Sw is in the line of the row's
// first advance or of the previous row's last one, else in a request of its own (a cycle without
// a chunk). A tag per request says what the line holds; the responses fill one scale entry per
// chunk ({Sw, the low block's data}, {the high block's data}) in the scale FIFO.
//
// Replay (ROWS > MCOLS, M > MCOLS): each streamed row is consumed G = ceil(M / MCOLS) times, once
// per group of MCOLS stationary rows (ACT RAM rows g*MCOLS ..). Group 0 pops the row's chunks from
// the FIFO without freeing them, the later groups read the same FIFO entries again (head + k) and
// the last group frees them, so a weight is streamed from DRAM once for all M rows (a row must fit
// the FIFO: KB <= DEPTH). Each (row, group) is a row of the pipeline and the drain; the drain steps
// its TMEM addresses by MCOLS * ors per group.
//
// Weight groups (WG > 1, docs/isa.md "Weight groups", docs/weight_groups.md): an MM with lg =
// log2 R (GRP) streams its operand in groups of R rows, WG = MCOLS (int8) or 2 WG (4-bit): a
// step's chunk is P = WG x D bytes from port W, K-block c of the group's rows, lane j column j's:
// int8 row j's D bytes, 4-bit rows 2j (the low nibbles) and 2j + 1 (the high), and every column
// takes it against one activation row m (column 0 from the ACT RAM's bank of row m, the others
// from their neighbour a cycle later: the activation hop). Each group is replayed for the MM's
// M activation rows as a row for its replay groups (passes m = 0 .. M-1), one advance a chunk.
// Scales come from port A, whose reads at WG > 1 answer with the AB = 128-byte beat holding the
// word (natural MMs take its 32-byte line): a group's Sw words (int8 one line, 4-bit two: with
// its first advance's request when they share a beat, else a request of its own before it),
// then the group's entries, a request a chunk (int8: 8 bytes of a line; int4x / fp4x6 two lines,
// NVFP4 four). The result row is the group's R integers against row m, with the rows' Sw and
// the row's Sa', drained to out + m*ors + g*R + r (rows past N not drained; 4-bit: two result
// FIFO rows, r 0..7 and 8..15, the second a cycle after the first); RMAX takes a result row's
// maximum into row m's running max, ASCALE row m's factor for all its results. At WG > 1 every
// MM is in groups (GALL, docs/isa.md "Every MM in groups"; a natural MM, lg 0, is refused): the
// chunk FIFO is P wide, column j's weights its lane j's, through its own skew; port B is unused.
// Port W's chunks come lane by lane (lane c: channel c's D bytes, column c's part: the images
// store each chunk's parts in the channel hash's order): each lane is written into the entry
// reserved at the request on its own, the entry is in the FIFO once every lane has written it,
// and column j reads lane j (the FIFO write side: docs/native_channels.md).
// Posted chunks (XPOST, WG > 1: the WG targets' (a) v2, docs/native_channels.md and docs/weight_groups.md): every
// chunk comes on port W with the FIFO entry it reserved (w_slot). The channels write the beats
// into the lanes' RAMs themselves, on their clock (p_clk; lane c channel c's, entry p_slot, 2^QB
// writes a beat), and count them (p_cnt, lane c's beats written, synchronized to clk by the
// adapter): an entry is in the FIFO once each lane has counted a beat past those of the entries
// before it, a lane's beats coming in its requests' order. With the lanes in UltraRAM
// (FIFO_STYLE "ultra", one clock) the channels write a staging RAM instead, and each counted beat
// is copied into its lane on clk (g_xp's g_u).
// Packed 4-bit (GRP at a 4-bit format, "pg"; docs/weight_groups.md): column j's
// PEs multiply the activation by w0 + w1 * 2^16, its lane's two rows' elements (the packed PE,
// otpu_pe PK: one multiplier), so a chain's sum is S0 + S1 * 2^16 with |S0|, |S1| <= 16 * 128 *
// 12 < 2^15; the chain's first PE adds 2^15, so the column end splits it by its bits: S0 =
// P[15:0] - 2^15 into the column end's multipliers with row 2j's B, S1 = P >> 16 into a second
// column end's with row 2j + 1's: column j's two sums are rows 2j and 2j + 1's integers.
// K tail (KT, WG > 1; docs/isa.md "K tail"): a GRP MM with rs bit 0 streams groups 2p and 2p + 1
// as one run of 2 KB - 1 chunks at sa + p (rs - 1): 2p's chunks 0 .. KB-2, their shared tail
// chunk T_p, 2p + 1's chunks 0 .. KB-2. The issuer requests no chunk for the odd group's first
// advance (T_p, the FIFO's head once the even group is done with it): the even group's last pass
// frees KB - 1 entries, keeping T_p, the odd group's frees KB (an odd count's last group its KB).
// The odd group's advances read ACT blocks ab + KB-1, ab, .., ab + KB-2, the first with its
// halves (byte i at i ^ D/2) and its record's sub-block offsets (da[b] at b ^ 2) swapped at column
// 0's input (g_bc), the hop carrying them on. Scale entries stay one an advance, in the image's
// order, so the scale FIFO holds up to one entry more than the chunk FIFO per pair (socc).
// Split terms (SPL, WG > 1; docs/isa.md "Split terms"): an MM with w7 bit31 takes a split
// product's terms as one row of KB advances: KB 3 hi.hi, lo.hi and hi.lo (ACT blocks ab, ab + 1
// and ab against the group's chunks at A, A and A + HLD), KB 2 hi and lo (ab against A, A + HLD);
// HLD = w7[30:20] KiB, the scale rows' stride w7[19:0]. The issuer steps the chunk address by 0
// or HLD instead of P, the ACT block's offset (ckb) is 1 on a triple's second advance, and the
// advances past the first are the lo terms, their W - dw 8 less (bdec's lo; RSC: dw 0); the
// row's Sa' is its last advance's, block ab's.
//
// Pumped array (PUMP, WG > 1; [uarch] MXU_PUMP): each column is an otpu_mxpc, its DSPs on clk_f =
// 2 x clk, which makes in a slow cycle the packed products of two activation rows against the
// column's chunk (B takes them on alternate fast edges, the cascade carries both rows' sums
// interleaved), so a GRP pass is rows 2p and 2p + 1 (the even bank's and the odd one's at the same
// ACT RAM address; an odd M's last pass row 2p alone): ceil(M / 2) passes a group (Gc; M <= 2:
// no replay). The chains' stages are half a slow cycle apart, so the skews are halved (the
// weights' ceil(k / 2), WASK's lane reads by wa_t), the rows' bytes are S0's reads (no AASK)
// skewed at column 0, and each column has two rows' column ends (u_ce / u_ce2 each; B from
// BRAW's raw bytes with both rows' offsets). A pass's result FIFO rows are pushed in the unpumped
// passes' order, a cycle each from its row end (g_prend), so the drain is the unpumped one's;
// a pass's first advance waits until its last will follow the last pass's rows (pu_hold).
// The MXU holds two commands: the issuer streams the newer one's chunks while the consumer
// finishes (and drains) the older one. The issuer yields DRAM ports to the DMA and QST
// (a_gnt/b_gnt). A row's first advance pops only once all of the row's chunks and scales are in
// the FIFOs, so a row never stalls in its middle and the array needs no clock enable.
module otpu_mxu
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int D     = 32,
  parameter int MCOLS = 8,
  parameter int ROWS  = MCOLS,   // ACT RAM rows: the most stationary rows of a command
  parameter int DEPTH = 16,
  parameter int LANES = 8,       // TMEM banks (the drain's lanes write distinct banks)
  parameter int TLANES = LANES,  // TMEM's banks (t_rben; its lanes, at least LANES)
  parameter int IMPL  = 2,       // the dot product: 2 systolic (the only one; CFG_UARCH reports it)
  parameter int CL    = 16,      // products per cascade chain (at most D/8)
  parameter int E     = 0,       // epilogue lanes (0: MCOLS / 2, at least 1; more than MCOLS: MCOLS)
  parameter int WG    = 1,       // weight rows a group: 1, or MCOLS = 8 (port W's P = WG x D bytes)
  parameter int AB    = 32,      // port A's read bytes: 32 (a line), at WG > 1 128 (a channel beat)
  parameter bit XPOST = 1'b0,    // WG > 1: port W's beats posted into the chunk FIFO (see the top)
  parameter int QB    = 0,       // XPOST: 2^QB writes a beat (0, or the bridges' aclk quarters: 2)
  parameter bit CHASH = 1'b0,    // the channels' hash (otpu_native_dram; unused here: every chunk
                                 //   is a group's, its lanes in the hash's order)
  parameter bit TQ    = 1'b0,    // reads not granted (TMEM replays them), writes through a queue,
                                 //   the epilogue's advance from flip-flops (the slice: TMEM_IMPL 3 RPL)
  parameter string RF_STYLE = "block", // the result FIFO's copies: "block" (read ahead) or
                                 //   "distributed" (LUT RAM; the WG targets', its LUTs before its block RAM)
  parameter int RF_ROWS = 0,     // the result FIFO's rows (0: 32 or 64, the rows in flight of
                                 //   one-advance rows; 32: one-advance rows at 32 / (LROW + 7))
  parameter bit BRAW  = 1'b1,    // WG > 1: the column ends' B from the scale entry's raw bytes
                                 //   skewed to them (g_col_end g_br; 0: B skewed, g_bs)
  parameter string FIFO_STYLE = "block", // the chunk FIFO's lanes: "block", or "ultra" (UltraRAM
                                 //   in 72-bit columns; XPOST: the beats staged across, g_xp)
  parameter bit IQ    = 1'b0,    // the accept decoupled from the issuer: an accepted command's
                                 //   issue waits in a queue of QD (iq) for the issuer, which no
                                 //   longer holds the next accept until it has issued the command
                                 //   before (docs/bcu1525_s2.md; simulation:
                                 //   also +mxiq)
  parameter int QD    = 2,       // commands queued (2..7): the head h, the next n and QD - 2 behind
                                 //   them (bq), each streaming once accepted (docs/native_channels.md)
  parameter bit WASK  = 1'b1,    // XPOST at UltraRAM lanes: the weights' chain-stage and column
                                 //   skews are the lanes' per-column read addresses (g_xp g_u's
                                 //   g_wa; 0: skew registers after the lane's read, g_wcol's g_s)
  parameter bit AASK  = 1'b1,    // WG > 1: column 0's activation skew is the ACT RAM's read by
                                 //   stage (otpu_actram ASK = CLS; g_bc g_ab; 0: skews, g_ask)
  parameter bit KT    = 1'b0,    // WG > 1: MM's K tail, rs bit 0 under GRP (see the top)
  parameter bit SPL   = 1'b0,    // WG > 1: MM SPL, a split's terms as one MM (w7 bit31; the top)
  parameter bit PUMP  = 1'b0,    // WG > 1: the array pumped on clk_f, two activation rows a pass
                                 //   (see the top; AASK off, BRAW on, D 128, chains of 16)
  parameter bit PHOLD = 1'b1,    // PUMP: the chain ends held into clk two fast cycles (otpu_mxpc)
  parameter int SID   = 0,
  parameter int DAW   = 32,      // DRAM byte address bits (otpu_seq AW): MM's sa and ssa (AW
  parameter int DWA   = waw(DAW) //   and WA here are WASK's); the ports' word address bits
) (
  input  logic                  clk,
  input  logic                  clk_f,       // PUMP: 2 x clk, rising with it; else unused
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
  output logic [MCOLS-1:0]      act_ren,     // per column: its ACT RAM read register advances
  input  logic [MCOLS*D*8-1:0]  act_data,
  input  logic [MCOLS*64-1:0]   act_scale,   // per column the record {da3, da2, da1, da0, Sa}
  // DRAM port A (scale lines) and B (chunks)
  output logic                  a_req,
  output logic [DWA-1:0]        a_addr,
  input  logic                  a_gnt,
  input  logic                  a_rvalid,
  input  logic [AB*8-1:0]       a_rdata,     // the aligned AB bytes holding word a_addr
  output logic                  b_req,
  output logic [DWA-1:0]        b_addr,
  input  logic                  b_gnt,
  input  logic                  b_rvalid,
  input  logic [D*8-1:0]        b_rdata,
  // DRAM port W (WG > 1: a grouped operand's P-byte chunks; unused at WG 1). Its beats come
  // lane by lane (lane c: the D bytes of channel c, otpu_native_dram; each lane's in its
  // requests' order, the lanes independent): lane c of a request holds the chunk's row c (the
  // images store each chunk's rows in the channel hash's order: docs/isa.md "Weight groups")
  output logic                  w_req,
  output logic [DWA-1:0]        w_addr,      // word address (P aligned)
  input  logic                  w_gnt,
  input  logic [WG-1:0]         w_bv,
  input  logic [WG*D*8-1:0]     w_bd,
  output logic [$clog2(DEPTH)-1:0] w_slot,   // XPOST: the FIFO entry the request reserved
  output logic                  w_one,       // XPOST: one beat (0: every MM in groups)
  // XPOST: the channels' beats, lane c channel c's, on p_clk: entry p_slot's quarter q, p_q
  // one-hot (bit q with p_we, else 0: each quarter RAM's write enable a register; QB 0: the
  // whole beat, p_q unused); the entry and p_q in copies of equal registers, one a RAM group
  // (p_slot: quarter RAM g's g, 0 .. 3, the remainder RAM's 4; p_q: the quarter RAMs' 0, the
  // remainder's field enables 1; each copy's fan-out its group's block RAMs); p_cnt: each
  // lane's beats written (mod 4 x DEPTH; clk's, from the adapter)
  input  logic                  p_clk,
  input  logic [WG-1:0]         p_we,
  input  logic [WG-1:0][1:0][((QB > 0) ? (1 << QB) : 1)-1:0] p_q,
  input  logic [WG-1:0][4:0][$clog2(DEPTH)-1:0] p_slot,
  input  logic [WG-1:0][(D*8 >> QB)-1:0] p_d,
  input  logic [WG-1:0][$clog2(DEPTH)+1:0] p_cnt,
  // TMEM (read port for ACC, write port)
  output logic [LANES-1:0]        t_ren,
  output logic [LANES-1:0][31:0]  t_raddr,
  input  logic [LANES-1:0][31:0]  t_rdata,
  output logic [LANES-1:0]        t_wen,
  output logic [LANES-1:0][31:0]  t_waddr,
  output logic [LANES-1:0][31:0]  t_wdata,
  input  logic                    t_gnt,
  input  logic                    t_lost,     // TQ: this cycle's read (or its replay) lost (replayed)
  // the read by TMEM bank (TLANES banks): bank b's enable and the address of the lane reading it
  // (t_ren / t_raddr's lanes merged per bank, as otpu_tmem's crossbar does; TQ: registers)
  output logic [TLANES-1:0]       t_rben,
  output logic [TLANES-1:0][31:0] t_rbad
);
  localparam int BW = $clog2(LANES);
  localparam int PW = $clog2(DEPTH);
  localparam int LM = 2, LA = 4;
  // epilogue lanes; NL: the drain's lanes per step
  // (a target's E holds at a smaller MCOLS too: the card's 4 is 2 at MCOLS 2)
  localparam int EL = (E > MCOLS) ? MCOLS : (E > 0) ? E : ((MCOLS >= 2) ? MCOLS / 2 : 1);
  localparam int NL = EL;
  // chains of CLS products, CPG of them per group of D/8 elements (1, or 2 above D = 128)
  localparam int CLS = (D / 8 < CL) ? D / 8 : CL;
  localparam int CPG = D / 8 / CLS;
  // S0 to the columns' row sums, deskewed (the column end's accumulator is valid CLS + 6 + j
  // edges after S0, column j deskewed by MCOLS - 1 - j)
  localparam int LROW = CLS + MCOLS + 5;
  // result FIFO rows: a row pops only while fewer than RF are between the pop and the drain, so
  // RF covers the pipeline's rows in flight (one-advance rows: one per cycle of its latency);
  // RF_ROWS fewer (a power of two, at least 16) throttles one-advance rows to RF in LROW + 7
  // cycles, as a LUT RAM copy of 32 rows is half one of 64
  localparam int RF = (RF_ROWS != 0) ? RF_ROWS : (LROW + 7 <= 32) ? 32 : 64;
  initial if (RF_ROWS != 0 && (RF_ROWS < 16 || (RF_ROWS & (RF_ROWS - 1)) != 0))
    $fatal(1, "otpu_mxu: RF_ROWS %0d (0, or a power of two >= 16)", RF_ROWS);
  localparam int RFW = $clog2(RF);
  localparam int MW = $clog2(MCOLS) + 1;
  localparam int TD = 128;                  // port A requests in flight (their tags)
  localparam int TDW = $clog2(TD);
  // a lane: the pick, i2f48 (4), x Sw, x Sa'
  localparam int LLAT = 1 + 4 + 2 * LM;
  // weight groups: a step's bytes, log2 WG, the column index bits
  localparam int P = WG * D;
  localparam int LGW = $clog2(WG);
  // GALL (WG > 1): every MM in groups, so no natural mode (port B, the natural decode and scale
  // reads, the columns' own activation skews past column 0, PAIR); WG 1: no groups
  localparam bit GA = (WG > 1);
  // PUMP (PU): the array's columns otpu_mxpc on clk_f, a pass two activation rows (2p, 2p + 1)
  localparam bit PU = PUMP && GA;
  initial if (PUMP && !GA) $fatal(1, "otpu_mxu: PUMP needs WG > 1");
  initial if (PU && (D != 128 || CL < 16 || !BRAW))
    $fatal(1, "otpu_mxu: PUMP needs D 128, chains of 16 and BRAW (D %0d, CL %0d)", D, CL);
  localparam bit AA = AASK && GA && !PU;
  // PU: a row's sums at the deskew's end (otpu_mxpc's row 0 at S0 + 15 + PHOLD + j, deskewed)
  localparam int LR = PU ? MCOLS + 14 + 32'(PHOLD) : LROW;
  initial if (AA && CLS < 2) $fatal(1, "otpu_mxu: AASK with chains of %0d (at least 2)", CLS);
  localparam bit KTE = KT && GA;            // (KT at WG 1: none)
  localparam bit SPE = SPL && GA;           // (SPL at WG 1: none)
  localparam bit CKB = KTE || SPE;          // the ACT block's offset a flip-flop (ckb)
  initial if (KTE && (D / 2) % CLS != 0)
    $fatal(1, "otpu_mxu: KT with D / 2 not a multiple of the chains' %0d", CLS);
  localparam int LCW = (MCOLS > 1) ? $clog2(MCOLS) : 1;
  // a chain's sum: 24 bits, or (WG > 1: the packed PE) 32
  localparam int PRW = (WG > 1) ? 32 : 24;
  initial if (IMPL != 2) $fatal(1, "otpu_mxu: IMPL 2 (systolic) is the only dot product");
  initial if (WG != 1 && (WG != MCOLS || WG != 8))
    $fatal(1, "otpu_mxu: WG 1, or WG = MCOLS = 8 (WG %0d, MCOLS %0d)", WG, MCOLS);
  initial if (XPOST && (WG == 1 || (QB != 0 && QB != 2)))
    $fatal(1, "otpu_mxu: XPOST needs WG > 1, and QB 0 or 2 (QB %0d)", QB);
  initial if (AB != ((WG > 1) ? 128 : 32))
    $fatal(1, "otpu_mxu: port A reads AB = 32 bytes at WG 1, 128 above (AB %0d)", AB);
  initial if (D % 16 != 0 || (D / 8) % CLS != 0 || CPG > 2)
    $fatal(1, "otpu_mxu: D a multiple of 16, D/8 a multiple of CLS, at most two chains a group");
  // the packed split needs one chain a group (two chains' S0 add up past 2^15)
  initial if (WG > 1 && CPG != 1) $fatal(1, "otpu_mxu: WG > 1 needs one chain a group (D/8 <= CL)");
  initial if (EL > LANES) $fatal(1, "otpu_mxu: E %0d > LANES", EL);
  initial if (ROWS < MCOLS || ROWS > 255) $fatal(1, "otpu_mxu: MCOLS <= ROWS < 256");

  // conflict-free drain run for a row stride: LANES / gcd(ors, LANES), at most NL
  function automatic logic [MW-1:0] drain_run(input logic [15:0] ors);
    int d;
    d = 1;                                   // ors a multiple of LANES: one lane per cycle
    for (int t = BW - 1; t >= 0; t--)        // the lowest set bit t: gcd = 2^t
      if (ors[t]) d = LANES >> t;
    return MW'((d < NL) ? d : NL);
  endfunction
  localparam logic [1:0] WF_W8 = 2'd0, WF_NV = 2'd3;
`ifndef SYNTHESIS
  // the checks start once the reset has held two edges (registers start arbitrary, and so does a
  // unit's reset register until the slice's reset pipeline fills: otpu_slice)
  bit rst_p, rst_seen;
  initial begin rst_p = 1'b0; rst_seen = 1'b0; end
  always @(posedge clk) begin rst_p <= rst; if (rst && rst_p) rst_seen <= 1'b1; end
  // a replayed row (M > MCOLS) must fit the FIFO: its chunks (4-bit: two blocks each)
  always @(posedge clk) if (!rst && rst_seen && start) begin
    if (int'(cmd.w6[23:16]) > ROWS) $fatal(1, "otpu_mxu: M %0d > ROWS %0d", cmd.w6[23:16], ROWS);
    if (int'(cmd.w6[23:16]) > MCOLS && ((cmd.flags[5:4] != WF_W8) ?
        (int'(cmd.w4[23:16]) + 1) / 2 : int'(cmd.w4[23:16])) > DEPTH)
      $fatal(1, "otpu_mxu: a replayed row (M > MCOLS) must fit the FIFO: KB %0d, DEPTH %0d",
             cmd.w4[23:16], DEPTH);
    if (cmd.flags[6] && cmd.flags[5:4] != WF_W8 && 2 * int'(cmd.w6[23:16]) > MCOLS)
      $fatal(1, "otpu_mxu: PAIR needs 2*M <= MCOLS");
    // groups of WG rows (int8) or 2 WG (4-bit: two rows a lane); at WG > 1 every MM's (GALL)
    if ((GA || cmd.lg != 4'd0) &&
        (WG == 1 || int'(cmd.lg) != LGW + ((cmd.flags[5:4] != WF_W8) ? 1 : 0)))
      $fatal(1, "otpu_mxu: MM in groups of 2^%0d rows, WG %0d, WF %0d (WG > 1: every MM's)",
             cmd.lg, WG, cmd.flags[5:4]);
    if (GA && cmd.flags[6]) $fatal(1, "otpu_mxu: groups take no PAIR");
    if (GA && (cmd.flags[2] || cmd.flags[3]) && cmd.w6[23:16] > 8'(MCOLS))
      $fatal(1, "otpu_mxu: RMAX and ASCALE need M <= MCOLS");
    // a replayed group (M > 1) must fit the FIFO: KB chunks
    if (WG > 1 && cmd.lg != 4'd0 && cmd.w6[23:16] > 8'd1 && int'(cmd.w4[23:16]) > DEPTH)
      $fatal(1, "otpu_mxu: a replayed group (M > 1) must fit the FIFO: KB %0d, DEPTH %0d",
             cmd.w4[23:16], DEPTH);
    // SPL (w7 bit31): KB 2 or 3, int8 RSC, k0 0, HLD > 0, not KT, RMAX or ASCALE
    if (SPE && cmd.w7[31] &&
        (cmd.w4[23:16] < 8'd2 || cmd.w4[23:16] > 8'd3 || cmd.flags[5:4] != WF_W8 ||
         !cmd.flags[7] || cmd.flags[0] || cmd.flags[2] || cmd.flags[3] || cmd.w4[31:24] != 8'd0 ||
         cmd.w5[0] || cmd.w7[30:20] == 11'd0))
      $fatal(1, "otpu_mxu: SPL with KB %0d, flags %h, k0 %0d, rs %h, HLD %0d", cmd.w4[23:16],
             cmd.flags, cmd.w4[31:24], cmd.w5, cmd.w7[30:20]);
    // K tail (rs bit 0 under GRP): k0 0, int8 / int4 / fp4, a pair stride of 2 KB - 1 chunks
    if (GA && cmd.lg != 4'd0 && cmd.w5[0]) begin
      if (!KT) $fatal(1, "otpu_mxu: rs bit 0 (KT) on an MXU without KT");
      if (cmd.w4[31:24] != 8'd0 || cmd.flags[5:4] == WF_NV)
        $fatal(1, "otpu_mxu: KT with k0 %0d, WF %0d (k0 0, not NVFP4)", cmd.w4[31:24],
               cmd.flags[5:4]);
      if (cmd.w4[23:16] != 8'd0 && 64'(cmd.w5) - 1 < 64'(2 * int'(cmd.w4[23:16]) - 1) * P)
        $fatal(1, "otpu_mxu: KT's pair stride %0d under (2 KB - 1) P, KB %0d", cmd.w5 - 1,
               cmd.w4[23:16]);
    end
  end
`endif

  // ================================================================== issuer
  logic        i_act, i_unit, i_w4, i_pair, i_rsc;
  logic        i_nv, i_pg;                  // NVFP4 under GRP; pg (packed)
  logic [PW-1:0] i_slot;                    // the next chunk request's FIFO entry (f_head + occ)
  logic        i_ini;                       // the command's first cycle: its addresses load
  logic [1:0]  i_wf;
  logic [7:0]  i_k0;
  logic [31:0] i_left, i_rs, i_srs;
  logic [15:0] i_KB, i_k;
  logic        i_klast, i_l1;               // i_k + 1 == i_KB; i_left == 1
  logic [DAW-1:0] row_addr, chunk_addr;    // (DAW bits; the strides i_rs, i_srs zero-extended)
  logic [DAW-1:0] srow_addr, srow_next;     // the row's scale row; the next row's
  logic [DAW-1:0] scale_addr;               // the advance's block data in the scale row
  logic [31:0] i_soff;                      // block k0's data offset in a scale row
  logic [7:0]  i_sstep;                     // the data address step per advance
  logic        i_swd;                       // the row's Sw came with an earlier line
  logic        i_swh;                       // the row's Sw is in the line of its first advance
                                            // (GRP: the group's in the beat of its first)
  logic        i_swn;                       // the next row's Sw is in the advance's line
  logic        i_sws;                       // a request for the row's Sw alone comes first
  logic        i_kt, i_odd;                 // KT: the command's; its group is a pair's second
  logic        i_spl;                       // SPL: the command's
  logic [20:0] i_hld;                       // SPL: HLD (lo's chunks after hi's), bytes
  logic        i_tk;                        // KT: the advance is the odd group's first (T_p, no
                                            // chunk request: the even group's request brought it)
  logic [PW:0] socc;                        // KT: scale entries issued and not yet freed
  logic [PW:0] occ;                         // chunks issued and not yet popped (<= DEPTH)
  logic [PW:0] occz;                        // WA: occ, its pops counted AW cycles late (the
                                            // issue's: no entry is reissued while read, g_fz)
  logic [TDW:0] tg_n;                       // port A requests in flight
  // The stream oracle (simulation only; the bound of a weight prefetch,
  // docs/bcu1525_s2.md): +mxrec=F writes each accepted command's issue, one line an accept with
  // chunks (F.<SID>); +mxpf=F replays such a list, each line with a gate, as the issuer's own:
  // the issuer streams entry i once the commands accepted exceed its gate (-1: at once), into the
  // FIFO as its space allows, whatever the sequencer has accepted; the commands keep the consumer's
  // order, each consuming its own entry's chunks (an accept checks it is the entry's command). A
  // reset (a CHAIN reload, every command done) rewinds it to the first entry not yet accepted.
  // Synthesis: pf_on 0, the issuer as before.
  logic        pf_on;
`ifdef SYNTHESIS
  assign pf_on = 1'b0;
`else
  bit          pf_b;
  int          pf_fd, pf_n, pf_i, pf_acc;
  int unsigned pf_tot [$], pf_kba [$], pf_w4 [$], pf_w5 [$], pf_w7 [$];
  int unsigned pf_fl [$], pf_nv [$], pf_pg [$], pf_pr [$];
  longint unsigned pf_a1 [$], pf_a2 [$];
  int          pf_g [$];
  initial begin
    string f;
    pf_b = 1'b0; pf_fd = 0; pf_n = 0; pf_i = 0; pf_acc = 0;
    if ($value$plusargs("mxrec=%s", f)) begin
      pf_fd = $fopen($sformatf("%s.%0d", f, SID), "w");
      if (pf_fd == 0) $fatal(1, "otpu_mxu: cannot write %s.%0d", f, SID);
    end
    if ($value$plusargs("mxpf=%s", f)) begin
      int fd, r, g;
      int unsigned t, k, nv, pg, pr, w4, w5, w7, fl;
      longint unsigned a1, a2;
      fd = $fopen($sformatf("%s.%0d", f, SID), "r");
      if (fd == 0) $fatal(1, "otpu_mxu: cannot read %s.%0d", f, SID);
      while (!$feof(fd)) begin
        r = $fscanf(fd, "%d %d %d %d %d %h %h %h %h %h %h %d\n", t, k, nv, pg, pr, w4, w5, w7, fl,
                    a1, a2, g);
        if (r != 12) break;
        pf_tot.push_back(t); pf_kba.push_back(k); pf_nv.push_back(nv); pf_pg.push_back(pg);
        pf_pr.push_back(pr); pf_w4.push_back(w4); pf_w5.push_back(w5); pf_w7.push_back(w7);
        pf_fl.push_back(fl); pf_a1.push_back(a1); pf_a2.push_back(a2); pf_g.push_back(g);
      end
      $fclose(fd);
      pf_n = pf_tot.size();
      pf_b = 1'b1;
      $display("otpu_mxu %0d: stream oracle, %0d entries (DEPTH %0d)", SID, pf_n, DEPTH);
    end
  end
  assign pf_on = pf_b;
  final if (pf_fd != 0) $fclose(pf_fd);
`endif
  // IQ: the issue of each accepted command not yet streamed, oldest first
  typedef struct packed {
    logic [31:0]    left, rs, srs;
    logic [15:0]    KB;
    logic [7:0]     k0;
    logic [1:0]     wf;
    logic           nv, pg, unit, rsc, w4, pair, kt, spl;
    logic [20:0]    hld;
    logic [DAW-1:0] a1, a2;
  } iss_t;
  iss_t        iq [QD];
  logic [$clog2(QD):0] iq_n;
  logic        iq_on;
`ifdef SYNTHESIS
  assign iq_on = IQ;
`else
  bit iq_b;
  initial iq_b = IQ || $test$plusargs("mxiq");
  assign iq_on = iq_b;
`endif

  // ================================================================== command queue (2)
  // GRP (every MM at WG > 1): the operand in groups of WG rows, or (pg: a 4-bit format) 2 WG;
  // pg's advance is the packed one, a chunk (one K-block of 16 rows) a cycle
  wire        cmd_pg = GA && cmd.flags[5:4] != WF_W8;
  // PAIR: 4-bit only, WG 1 only; a row is ceil(KB/2) advances of one chunk each. KB is 8 bits,
  // k0 above it
  wire        cmd_pair = cmd.flags[6] && cmd.flags[5:4] != WF_W8 && !GA;
  wire [7:0]  cmd_KB8 = cmd.w4[23:16];
  wire [15:0] cmd_KBa = cmd_pair ? (16'(cmd_KB8) + 16'd1) >> 1 : 16'(cmd_KB8);
  // the rows streamed: N, or GRP's groups ceil(N / R); the last group's rows (1 .. R)
  localparam int LGP = LGW + 1;             // log2 of a 4-bit group's rows
  wire [15:0] cmd_N = !GA ? cmd.w4[15:0]
                    : cmd_pg ? 16'((32'(cmd.w4[15:0]) + 32'(2 * WG) - 1) >> LGP)
                             : 16'((32'(cmd.w4[15:0]) + 32'(WG) - 1) >> LGW);
  wire [LGP:0] cmd_Nr = (LGP+1)'(32'((cmd.w4[15:0] - 16'd1) & 16'((cmd_pg ? 2 * WG : WG) - 1))
                                 + 32'd1);
  wire [31:0] cmd_total = 32'(cmd_N) * 32'(cmd_KBa);   // advances (chunk requests)
  wire        cmd_nv = GA && cmd.flags[5:4] == WF_NV;
  // SPL: the scale rows' stride w7[19:0] (HLD above it)
  wire        cmd_spl = SPE && cmd.w7[31];
  wire [31:0] cmd_srs = cmd_spl ? {12'd0, cmd.w7[19:0]} : cmd.w7;
  // cmd_total == 0 without the multiply (a product of two factors is zero iff one is)
  wire        cmd_tz = (cmd.w4[15:0] == 16'd0) || (cmd_KB8 == 8'd0);
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && rst_seen && start && cmd_tz != (cmd_total == 0))
      $fatal(1, "otpu_mxu: cmd_tz %0d but total %0d", cmd_tz, cmd_total);
`endif
  // The two entries are the head (h) and the next command (n), not two slots and a head pointer:
  // every head field is its own flop, with no 2:1 mux on a pointer in front of the consumer and
  // the drain. Completing the head swaps them (n keeps the old head, as the idle slot did).
  typedef struct packed {
    logic [31:0]      out, total;
    logic             tz;              // total == 0 (registered: off the drain path)
    logic [15:0]      KB, KBa;         // KBa: advances per row
    logic [15:0]      KBm1;            // KBa - 1: the last advance's ck (last_k a compare, no adder)
    logic             pair;
    logic [MCOLS-1:0] hi;              // PAIR: the rows (banks) that read the odd blocks (j >= M)
    logic [31:0]      mxo;             // RMAX output base: out + M * ors (no multiply later)
    logic [7:0]       M, ab;
    logic [MW-1:0]    run;             // drain lanes per cycle without a bank conflict
    logic             unit, acc, rmax, asc, go, rsc;
    logic [1:0]       wf;
    logic [31:0]      asa;
    logic [MCOLS-1:0][31:0] jo;        // j * ors
    logic [7:0]       G;               // groups: ceil(M / MCOLS)
    logic [MW-1:0]    Ml;              // rows of the last group
    logic [31:0]      gs;              // drain address step to the next group (MCOLS * ors)
    logic             g1;              // G == 1: a row's first group is its last
    logic [MW-1:0]    dnl;             // the last group's first drain step: its lanes (min(run,
    logic             dfl;             // Ml)) and whether it drains the group (Ml <= run)
    logic [MCOLS-1:0] cm;              // the columns its advances read (act_ren)
    // GRP (WG > 1): the replay groups are its M activation rows (G = M), each result row a weight
    // group's WG results (the last group's Nl), at out + m*ors + g*WG + j; pg: two result rows
    // a pass, rows 0..7 and 8..15 of its 16 (the last group's Nl and Nl1 of them)
    logic             pg;
    logic [15:0]      Gw;              // weight groups
    logic [MW-1:0]    Nl, Nl1;         // the last weight group's rows (results; pg: per half)
    logic [MW-1:0]    dnlg, dnlg1;     // that group's first drain steps (as dnl, dfl for Nl, Nl1)
    logic             dflg, dflg1;
    logic             kt;              // KT: a pair of groups shares its tail chunk (rs bit 0)
    logic             spl;             // SPL: a split's terms (w7 bit31)
    logic             t1;              // total == 1 (c_l1's load)
    logic [7:0]       Gc;              // the consumer's passes: G (PU: ceil(M / 2), rows 2p and
    logic             Mo;              //   2p + 1 a pass); PU: M odd (the last pass has no row 1)
  } qent_t;
  qent_t       h, n;                    // the head, the next command
  qent_t       cmd_e;                   // cmd as an entry (not yet released)
  // QD > 2: the commands behind n, oldest first (bq[0]). They only wait: the consumer and the
  // drain read h, the pop h or n, so a command reaches the pipeline through n as at QD 2, and
  // the issuer streams each one from its accept (its chunks in order behind the others').
  localparam int BQD = (QD > 2) ? QD - 2 : 1;
  logic [2:0]  q_n;
  qent_t       bq [BQD];
  logic [2:0]  bq_c;                    // entries in bq (q_n - 2 when q_n > 2)

  always_comb begin
    int g;
    cmd_e = '0;
    cmd_e.out   = cmd.w3;
    cmd_e.total = cmd_total;
    cmd_e.tz    = cmd_tz;
    cmd_e.KB    = 16'(cmd_KB8);
    cmd_e.KBa   = cmd_KBa;
    cmd_e.KBm1  = cmd_KBa - 16'd1;
    cmd_e.pair  = cmd_pair;
    for (int j = 0; j < MCOLS; j++)
      cmd_e.hi[j] = cmd_pair && 32'(j) >= 32'(cmd.w6[23:16]);
    cmd_e.mxo   = cmd.w3 + 32'(cmd.w6[23:16]) * 32'(cmd.w6[15:0]);
    cmd_e.M     = cmd.w6[23:16];
    cmd_e.run   = drain_run(cmd.w6[15:0]);
    cmd_e.ab    = cmd.w6[31:24];
    cmd_e.unit  = cmd.flags[0];
    cmd_e.rsc   = cmd.flags[7] && !cmd.flags[0];
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
    // the ACT RAM columns its advances read: every column with replay (G > 1), else rows j < M
    // (PAIR: and the high rows j < 2M); the others are no output's
    for (int j = 0; j < MCOLS; j++)
      cmd_e.cm[j] = (g > 1) || 32'(j) < (cmd_pair ? 32'd2 : 32'd1) * 32'(cmd.w6[23:16]);
    if (GA) begin
      // a result row's results are consecutive words (any stride's run is 1's); the passes step
      // by ors (pg: from the second result row's, ors - MCOLS), the weight groups by R; every
      // column reads (the advance's row's bank)
      cmd_e.pg    = cmd_pg;
      cmd_e.run   = drain_run(16'd1);
      for (int j = 0; j < MCOLS; j++) cmd_e.jo[j] = 32'(j);
      cmd_e.G     = (cmd.w6[23:16] == 8'd0) ? 8'd1 : cmd.w6[23:16];
      cmd_e.Ml    = MW'(MCOLS);
      cmd_e.gs    = 32'(cmd.w6[15:0]) - (cmd_pg ? 32'(MCOLS) : 32'd0);
      cmd_e.g1    = (cmd.w6[23:16] <= 8'd1);
      cmd_e.dnl   = (cmd_e.run < cmd_e.Ml) ? cmd_e.run : cmd_e.Ml;
      cmd_e.dfl   = (cmd_e.Ml <= cmd_e.run);
      cmd_e.cm    = '1;
      cmd_e.Gw    = cmd_N;
      cmd_e.Nl    = (32'(cmd_Nr) > MCOLS) ? MW'(MCOLS) : MW'(cmd_Nr);
      cmd_e.Nl1   = (32'(cmd_Nr) > MCOLS) ? MW'(32'(cmd_Nr) - MCOLS) : '0;
      cmd_e.dnlg  = (cmd_e.run < cmd_e.Nl) ? cmd_e.run : cmd_e.Nl;
      cmd_e.dflg  = (cmd_e.Nl <= cmd_e.run);
      cmd_e.dnlg1 = (cmd_e.run < cmd_e.Nl1) ? cmd_e.run : cmd_e.Nl1;
      cmd_e.dflg1 = (cmd_e.Nl1 <= cmd_e.run);
      cmd_e.kt    = KTE && cmd.lg != 4'd0 && cmd.w5[0];
      cmd_e.spl   = cmd_spl;
      cmd_e.Gc    = (PU && cmd.w6[23:16] > 8'd1) ? 8'((9'(cmd.w6[23:16]) + 9'd1) >> 1) : cmd_e.G;
      cmd_e.Mo    = PU && cmd.w6[16];
    end
    cmd_e.t1    = (cmd_N == 16'd1) && (cmd_KBa == 16'd1);
    if (!GA) cmd_e.Gc = cmd_e.G;
  end

  // Two heads (docs/mxu_systolic.md): the drain's head is h, the oldest command: its results are
  // drained, its completion signalled. The pop head is h, or n once the head's chunks have all
  // popped (pn), if the pipeline treats n alike (the same PAIR and M, the per-command state read
  // after S0): the next command's rows then enter the pipeline behind the head's instead of after
  // its drain. pop_q is the pop head's entry, a register loaded with h and n (pn ? n : h), so
  // neither the pop nor the drain reads its command through a mux.
  logic        pn;
  qent_t       pop_q, pop_e;
  assign pop_e = pop_q;
  // pop_q's format, a kept copy loaded as pop_q is, for wf0 alone: pop_q.wf drives the column
  // copies and the pop's controls (fo 73), and its wire to wf0 beside the decodes was B1's
  // worst MXU path (pop_q[wf] -> wf0, 177 rows); the copy lets the placer split it
  (* dont_touch = "true" *) logic [1:0] pwf;

  wire [31:0] c_out = h.out;
  wire        c_tz = h.tz;
  // the pop head's (c_KB .. c_wf, p_*) and the drain head's (the rest)
  wire [15:0] c_KB = pop_e.KB, c_KBa = pop_e.KBa;
  wire        c_pair = pop_e.pair;
  wire [MCOLS-1:0] c_hi = pop_e.hi;
  wire [7:0]  c_M = h.M, c_ab = pop_e.ab;
  wire [7:0]  p_M = pop_e.M, p_G = pop_e.Gc;     // (the consumer's passes)
  wire        p_act = (q_n != 0) && pop_e.go;
  wire [MW-1:0] c_run = h.run;
  wire        c_unit = pop_e.unit, c_rsc = pop_e.rsc, c_acc = h.acc, c_rmax = h.rmax;
  wire        c_pg = pop_e.pg;                      // the pop head's pg (0 at WG 1)
  wire        c_1a = c_pair || c_pg;                // 4-bit: one advance a chunk (pg: a block)
  wire        c_asc = h.asc;
  wire [1:0]  c_wf = pop_e.wf;
  wire        c_w4 = (c_wf != WF_W8);
  wire [31:0] c_asa = h.asa;
  wire        c_act = (q_n != 0) && h.go;
  wire [7:0]  c_G = h.G;
  logic [31:0] alpha [MCOLS];
  logic [1:0]  al_st;                       // ASCALE factors: 0 to load, 1 requested (in TMEM's
                                            // request register), 3 read (the words arrive), 2 loaded
  logic [7:0]  al_i, mx_i;                  // next ASCALE factor to load / RMAX value to write

  // FIFOs of chunks and of their scale entries (one per chunk; the two DRAM ports return
  // independently, in order; an issued chunk's slot is reserved, so neither FIFO can overflow)
  // block RAM: written in their own reset-free process below. The chunk FIFO is its own module
  // (kept as a hierarchy): inline, Vivado absorbed its read register into the DSP input
  // registers of the products, which left an asynchronous read.
  logic [PW-1:0]  f_head, f_tail;
  logic [PW-1:0]  s_head, s_tail;
  logic [PW:0]    f_count, s_count;
  // FIFO read addresses, kept equal to (last group ? head : head + entry k of the row) by the
  // pops: a register, so no adder sits in front of the block RAM address
  logic [PW-1:0]  f_rd, s_rd;
  // The chunk FIFO is FG block RAM column groups (whole 36-bit BRAM columns each; FIFO_STYLE
  // "ultra": 72-bit URAM288 columns), each with its own copies of the write and read addresses
  // (f_tl, f_rdl: f_tail's and f_rd's updates): one register no longer drives every RAM of the
  // chunk across its span. f_tail and f_rd remain for the checks.
  localparam bit FU = FIFO_STYLE == "ultra";
  localparam int FW = D * 8;
  localparam int FB = FU ? 72 : 36;         // a RAM column's bits
  localparam int FC = (FW + FB - 1) / FB;   // RAM columns
  localparam int FG = (FC < 4) ? FC : 4;
  function automatic int fg_lo(input int g);
    int c;
    c = 0;
    for (int i = 0; i < g; i++) c = c + FC / FG + ((i < FC % FG) ? 1 : 0);
    return (FB * c < FW) ? FB * c : FW;
  endfunction
  (* keep *) logic [FG-1:0][PW-1:0] f_tl, f_rdl;
  // WASK (WA: XPOST, UltraRAM lanes): a lane's FC columns hold its D bytes in stage-major order,
  // SPC a column: slot k is chain stage k / BPS's byte k % BPS, the lane's byte wa_byte(k), so a
  // column's bytes are one or two adjacent stages' and the column is read when its PEs need its
  // first stage, wa_sm (g_xp g_u's g_wa). wsk: each lane's bytes at their decode's cycle (the
  // skews' ends); AW: the cycles a pop's reads trail it by at most, plus margin (the issuer does
  // not reuse an entry freed within them: occz)
  localparam bit WA = WASK && XPOST && FU;
  localparam int SPC = FB / 8;
  localparam int BPS = D / CLS;
  function automatic int wa_byte(input int k);
    return k / BPS + CLS * (k % BPS);
  endfunction
  function automatic int wa_sm(input int m);
    return SPC * m / BPS;
  endfunction
  // a stage's cycle (after the lane's first): stage s's, or (PU: otpu_mxpc's A2 / D) ceil(s / 2)
  function automatic int wa_t(input int s);
    return PU ? (s + 1) / 2 : s;
  endfunction
  localparam int AW = WA ? wa_sm(FC - 1) + MCOLS + 1 : 0;
  logic [MCOLS-1:0][D-1:0][7:0] wsk;
  if (!WA) begin : g_nwsk
    assign wsk = '0;
  end

  // ================================================================== consumer control
  logic [15:0] ck;
  logic [7:0]  cg;                          // the group of the row being consumed (replay)
  logic [31:0] c_left;                      // chunks of the pop head not yet freed
  // c_left == 0, a flip-flop loaded with c_left (an entry's tz, or c_left == 1 at its decrement):
  // the drain's test (c_drained -> c_done -> the drain lanes' address enables) and the pop's
  // (more) start at it, not at a 32-bit compare of c_left (fanout 136; the m8 card at 150 MHz:
  // c_left -> c_drained -> dal's enables, 8 levels, 5.7 ns of route, -0.080 ns)
  logic        c_lz;
  // a command accepted last cycle became the head: c_left takes its total now, from h (a flop),
  // not from cmd_total in the start cycle. It cannot pop yet (released a cycle after its start at
  // the earliest), and nothing may treat the stale c_left == 0 as drained meanwhile
  logic        cl_ld;
  logic        c_l1;                        // c_left == 1 (KT: the command's last advance), as c_lz
  // KT: the pop head's group is a pair's second (c_odd), and the advance's ACT block from c_ab
  // (ckb: ck, an odd group's KB-1, 0, .., KB-2), both flip-flops kept with ck
  logic        c_odd;
  logic [15:0] ckb;
  logic [RFW:0] rows_live;                  // the head's rows popped (first block), not drained
  logic [RFW:0] rows_p;                     // the pop head's, while it is not the head
  wire last_k   = (ck == pop_e.KBm1);         // (ck + 1 == c_KBa)
  wire c_kt     = KTE && pop_e.kt;
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && rst_seen && p_act && last_k != (32'(ck) + 1 == 32'(c_KBa)))
    $fatal(1, "otpu_mxu: last_k %0d, ck %0d, KBa %0d", last_k, ck, c_KBa);
  bit cl_seen;                                 // a reset seen (registers start arbitrary)
  initial cl_seen = 1'b0;
  always @(posedge clk) if (rst) cl_seen <= 1'b1;
  always @(posedge clk) if (!rst && rst_seen && cl_seen && c_lz != (c_left == 0))
    $fatal(1, "otpu_mxu: c_lz %0d, c_left %0d", c_lz, c_left);
  always @(posedge clk) if (!rst && cl_seen && c_l1 != (c_left == 1))
    $fatal(1, "otpu_mxu: c_l1 %0d, c_left %0d", c_l1, c_left);
  always @(posedge clk) if (!rst && p_act && CKB &&
                            ckb != ((SPE && pop_e.spl) ? {15'd0, ck[0] && pop_e.KB[0]} :
                                    !c_odd ? ck : (ck == 0) ? pop_e.KBm1 : ck - 16'd1))
    $fatal(1, "otpu_mxu: ACT block offset %0d at advance %0d (odd group %0d, KB %0d)", ckb, ck,
           c_odd, c_KBa);
`endif
  wire last_g   = (cg + 1 == p_G);          // the row's last group: its pops free FIFO entries
  wire more     = p_act && !c_lz;
  // the chunk of advance ck (4-bit without PAIR or pg: two advances a chunk) and whether the
  // advance finishes it
  wire [15:0] ckc = (c_w4 && !c_1a) ? {1'b0, ck[15:1]} : ck;
  wire cdone    = !c_w4 || c_1a || ck[0] || last_k;
  // advance ck's chunk and scale entry are in the FIFOs: the next entries (one group), or entry
  // ckc of the row (group 0 of several; the later groups find the whole row)
  wire f_av     = (p_G == 8'd1) ? (f_count != 0) : (cg != 0 || 32'(f_count) > 32'(ckc));
  wire s_av     = (p_G == 8'd1) ? (s_count != 0) : (cg != 0 || 32'(s_count) > 32'(ckc));
  // pop: one block (PAIR: one chunk) advances into the pipeline; fpop: its chunk leaves the FIFO
  // (4-bit without PAIR: after the high half, or after the row's last block; replay: in the row's
  // last group). A row's first advance pops only once all of the row's chunks and scale entries
  // are in the FIFOs (a later group finds them there), so a row never stalls in its middle: a
  // cycle without a pop is a bubble between rows
  wire [15:0] c_rch = (c_w4 && !c_1a) ? 16'((32'(c_KBa) + 1) / 2) : c_KBa;   // chunks a row
  wire row_in   = cg != 0 ||
                  (32'(f_count) >= 32'(c_rch) && (c_unit || 32'(s_count) >= 32'(c_rch)));
  // pg: a row is two result FIFO rows (rows_live and rows_p count them), the second pushed a
  // cycle after the first, so a row's last advance never follows another's (KB 1: a bubble)
  logic lk1;                                // the last cycle popped a pg row's last advance
  // PU: a pass is rows 2p and 2p + 1 (none past an odd M: the last pass's row 1), its result FIFO
  // rows c_np (pg: two each), pushed a cycle apart from its row end on, so a pass's last advance
  // pops at least the last pass's rows' cycles after that one's. A pass never stalls once begun:
  // its first advance waits (lq: the rows of the passes that ended 1, 2 and 3 cycles ago; km
  // the pass's advances after its first, at most 4)
  wire c_x1     = PU && !(last_g && pop_e.Mo);
  wire [2:0] c_np = 3'(c_pg ? 2 : 1) << (c_x1 ? 1 : 0);
  logic [2:0] lq1, lq2, lq3;
  wire [3:0] km = (pop_e.KBm1 > 16'd4) ? 4'd4 : 4'(pop_e.KBm1);
  wire pu_hold  = ck == 16'd0 && (4'(lq1) > 4'd1 + km || 4'(lq2) > 4'd2 + km ||
                                  4'(lq3) > 4'd3 + km);
  wire pg_hold  = PU ? pu_hold : (WG > 1) && c_pg && last_k && lk1;
  wire pop      = more && f_av && (c_unit || s_av) && !pg_hold &&
                  (ck != 0 || (row_in && (RFW+2)'(rows_live) + (RFW+2)'(rows_p) +
                                         (PU ? (RFW+2)'(c_np - 3'd1) : (RFW+2)'(c_pg)) <
                                         (RFW+2)'(RF)));
  wire fpop     = pop && last_g && cdone;
  // KT: an even group's last advance keeps its chunk (T_p, the odd group's first) unless it is
  // the command's last (ktt): the chunk FIFO frees on fpop_f, the scale FIFO on fpop
  wire ktt      = c_kt && !c_odd && !c_l1 && last_k;
  wire fpop_f   = fpop && !ktt;
  logic fpz;                                // fpop_f as the issuer frees it (WA: AW cycles late)
  if (AW > 0) begin : g_fz
    logic [AW-1:0] fq;
    always_ff @(posedge clk) fq <= rst ? '0 : {fq[AW-2:0], fpop_f};
    assign fpz = fq[AW-1];
  end else begin : g_nfz
    assign fpz = fpop_f;
  end
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && rst_seen) begin
    if (more && ck != 0 && !pop) $fatal(1, "otpu_mxu: row stalled at advance %0d", ck);
    if (more && 32'(c_rch) > DEPTH) $fatal(1, "otpu_mxu: a row (%0d chunks) must fit the FIFO", c_rch);
  end
  always @(posedge clk) if (!rst && rst_seen && more) begin
    if (f_rd != (last_g ? f_head : f_head + PW'(ckc)))
      $fatal(1, "otpu_mxu: FIFO read address %0d, head %0d, k %0d", f_rd, f_head, ck);
    if (!c_unit && s_rd != (last_g ? s_head : s_head + PW'(ckc)))
      $fatal(1, "otpu_mxu: scale read address %0d, head %0d, k %0d", s_rd, s_head, ck);
  end
`endif
  // The issuer walks advances: a chunk request with every 8-bit block, every even 4-bit block (an
  // odd one's chunk is already on its way, so it needs no FIFO slot) and every PAIR chunk; a scale
  // line request with each advance (RSC: with each chunk), and the row's Sw line first when it is
  // in neither neighbour's line (i_sws: a cycle of its own, port A only).
  // GRP (WG > 1): the chunk (from port W) with every advance as natural ones (pg's as PAIR's), a
  // scale request (a beat) with every advance, the group's Sw line in its first advance's beat
  // or a request of its own (i_sws) before it
  // KT: the odd group's first advance requests no chunk (i_tk), and an advance's scale entry
  // needs a free one of its own (socc: the scale FIFO holds a pair's T_p entries twice)
  wire adv_b    = !i_w4 || i_pair || i_pg || !i_k[0];
  wire need_b   = !i_sws && adv_b && !GA;
  wire need_w   = !i_sws && adv_b && GA && !(KTE && i_tk);
  wire need_a   = !i_unit && (i_sws || !i_rsc || adv_b);
  wire want_iss = i_act && (!(need_b || need_w) || (WA ? occz : occ) < (PW+1)'(DEPTH)) &&
                  (!need_a || !tg_n[TDW]) &&
                  (!KTE || i_sws || i_unit || socc < (PW+1)'(DEPTH));
  wire go_iss   = want_iss && (!need_b || b_gnt) && (!need_w || w_gnt) && (!need_a || a_gnt);
  // the step ends its chunk (the chunk address moves on)
  wire end_c    = !i_w4 || i_pair || i_pg || i_k[0];
  // the tag of this cycle's port A request
  typedef struct packed {
    logic [1:0] wf;
    logic       pair, rsc;
    logic [4:0] doff;          // the block data's byte in the line
    logic [1:0] hv;            // the entry's halves it writes (01 low, 10 high, 11 both)
    logic       ent;           // it writes the entry (data, or RSC's Sw)
    logic       cpl;           // ... and completes it
    logic       swe;           // the entry's Sw is this line's word swo
    logic [2:0] swo;
    logic       swr;           // the next rows' Sw is this line's word sro (Sw-only: the row's)
    logic [2:0] sro;
    logic [1:0] lx;            // (AB > 32) the request's line in the beat
    logic [1:0] sl;            // (AB > 32) the Sw line's (swe: GRP's Sw line, RSC's entry)
    logic [3:0] wq;            // GRP (WG > 1): a group's beat (swr: its Sw lines; ent: the entry's
                               //   quarters wq, the beat's lines lx + q: int8 quarter 0 (8 bytes
                               //   of line lx), int4x / fp4x6 0 and 1, NVFP4 all four)
  } atag_t;
  atag_t a_tag;
  always_comb begin
    a_tag = '0;
    a_tag.wf = i_wf; a_tag.pair = i_pair; a_tag.rsc = i_rsc;
    a_tag.doff = scale_addr[4:0];
    a_tag.swo = srow_addr[4:2];
    a_tag.sro = i_sws ? srow_addr[4:2] : srow_next[4:2];
    a_tag.lx = (i_sws || i_rsc) ? srow_addr[6:5] : scale_addr[6:5];
    a_tag.sl = srow_addr[6:5];
    if (i_sws) begin
      a_tag.swr = 1'b1;
    end else if (i_rsc) begin
      a_tag.ent = 1'b1; a_tag.cpl = 1'b1; a_tag.swe = 1'b1;
    end else begin
      a_tag.ent = 1'b1;
      a_tag.hv = (i_pair || !i_w4) ? (i_pair ? 2'b11 : 2'b01) : (i_k[0] ? 2'b10 : 2'b01);
      a_tag.cpl = !i_w4 || i_pair || i_k[0] || i_klast;
      a_tag.swe = (i_k == 0) && !i_swd && i_swh;
      a_tag.swr = i_klast && !i_l1 && i_swn;
    end
    // GRP: each step a beat with the chunk's entries (a block's: int8 8 bytes of a line, int4x /
    // fp4x6 two lines, NVFP4 four), which it completes; the group's Sw lines come with its first
    // step's beat (swe, lines sl, sl + 1) or before it in a request of their own (i_sws)
    if (GA) begin
      a_tag.wq = !i_w4 ? 4'b0001 : i_nv ? 4'b1111 : 4'b0011;
      if (!i_sws && !i_rsc) begin
        a_tag.cpl = end_c || i_klast;
        a_tag.swe = (i_k == 0) && i_swh;
        a_tag.swr = 1'b0;
      end
    end
  end

  assign rdy = (pf_on || iq_on || (!i_act && !i_ini)) && (q_n < QD);
  assign computing = pop;
  assign pf_level  = f_count;
  assign pf_starve = more && f_count == 0;
  assign pf_block  = more && f_count != 0 && !pop;
  assign b_req  = go_iss && need_b;
  // not gated with b_req: the address (and the slice's parity of it) does not wait for b_gnt
  assign b_addr = chunk_addr >> 2;
  assign w_req  = go_iss && need_w;
  assign w_addr = chunk_addr >> 2;
  assign w_slot = XPOST ? i_slot : '0;
  assign w_one  = 1'b0;                     // (every chunk a group's: all its lanes)
  assign a_req  = go_iss && need_a;
  assign a_addr = !a_req ? '0 : (i_sws || i_rsc) ? (srow_addr >> 2) : (scale_addr >> 2);
  assign act_blk = 16'(c_ab) + (CKB ? ckb : c_pair ? {ck[14:0], 1'b0} : ck);   // (CKB: no PAIR)
  // GRP: the advance's activation row (column 0 reads its bank's): the pass's row cg (PU: rows
  // 2 cg, an even bank's, and 2 cg + 1, the next bank's at the same address)
  wire [7:0] act_row = PU ? {cg[6:0], 1'b0} : cg;
  // past an odd KB the high block is not the MM's (block ab + KB, outside its ACT range: a later
  // QACT may be writing it, and the scoreboard does not hold that one back): the high rows read
  // the even block again, which their zero multipliers (m0.o low) ignore
  // (both blocks' addresses from flip-flops, last_k picking at the end: it was the adder's carry in,
  // and the address goes on to the ACT RAMs' block RAM pins, the m8 card's -0.044 ns at 150 MHz)
  wire [15:0] act_b0 = 16'(c_ab) + {ck[14:0], 1'b0}, act_b1 = 16'(c_ab) + {ck[14:0], 1'b1};
  assign act_blk2 = (last_k && c_KB[0]) ? act_b0 : act_b1;
  assign act_hi = c_hi;
  assign act_grp = GA ? 8'(act_row >> LCW) : cg;
`ifndef SYNTHESIS
  // co-sim statistic of the scale reads: port A line requests, and those that fetch a row's Sw
  // alone (i_sws: its Sw in neither neighbour's line). Totals in xs_a / xs_sw (otpu_slice prints
  // them at the dump: rtlsim's stats["mxu_scale"]); under +trace one line a command as its
  // issue ends, "MXS<sid> a=<requests> sw=<Sw alone>", in issue order (= the MMs' order;
  // tools/perf_qwen.py's MM table). Not a trace event: the board does not record it.
  longint unsigned xs_a, xs_sw;
  int unsigned     xc_a, xc_sw;
  bit              xs_trace;
  initial begin
    xs_a = 0; xs_sw = 0; xc_a = 0; xc_sw = 0;
    xs_trace = $test$plusargs("trace");
  end
  always @(posedge clk)
    if (rst) begin
      xc_a = 0; xc_sw = 0;
    end else if (go_iss) begin
      if (a_req) begin
        xs_a++; xc_a++;
        if (i_sws) begin xs_sw++; xc_sw++; end
      end
      if (!i_sws && i_left == 1) begin
        if (xs_trace) $display("MXS%0d a=%0d sw=%0d", SID, xc_a, xc_sw);
        xc_a = 0; xc_sw = 0;
      end
    end
`endif
  // The ACT RAM's read enables (one per column): an advance reads its block in the cycle it pops
  // (S0 takes the read register's output), and only the columns of an output row read it. A
  // column's enable is the pop head's cm, gated with the head being released (p_act: every pop
  // has it), so a command of M < MCOLS rows (decode: M = 1) reads M of the MCOLS columns, and no
  // column reads while no command is released. Both are registers (pop_q, q_n) one LUT from the
  // block RAMs' enables: pop itself is not on that path. A column that does not read holds its
  // last block, which only its own products (no output's) take.
  assign act_ren = {MCOLS{p_act}} & pop_e.cm;
`ifndef SYNTHESIS
  // every column of an output row reads its block when the advance pops: rows j < Ml of a
  // command's last group (PAIR: and its high rows j < 2M), every row of the earlier groups
  always @(posedge clk)
    if (!rst && rst_seen && pop)
      for (int j = 0; j < MCOLS; j++)
        if (!act_ren[j] && (!last_g || 32'(j) < (c_pair ? 32'd2 : 32'd1) * 32'(pop_e.Ml)))
          $fatal(1, "otpu_mxu: column %0d of an output row does not read its ACT block (M %0d)",
                 j, p_M);
  // co-sim statistic of the ACT RAM reads (otpu_slice prints them at the dump: rtlsim's
  // stats["act_reads"]): cycles, column reads (enabled), pops, and the columns of output rows at
  // the pops, and the columns that ever read. Each column read is D bytes and an 8-byte scale
  // record.
  longint unsigned xa_cyc, xa_rd, xa_pop, xa_use;
  logic [MCOLS-1:0] xa_any;
  initial begin xa_cyc = 0; xa_rd = 0; xa_pop = 0; xa_use = 0; xa_any = '0; end
  always @(posedge clk)
    if (!rst) begin
      xa_cyc++;
      xa_rd += longint'($countones(act_ren));
      xa_any |= act_ren;
      if (pop) begin
        xa_pop++;
        for (int j = 0; j < MCOLS; j++)
          if (!last_g || 32'(j) < (c_pair ? 32'd2 : 32'd1) * 32'(pop_e.Ml)) xa_use++;
      end
    end
`endif

  // ---- the tags of the requests in flight (LUT RAM) and the responses: a response is
  // registered with its tag (R1), then written into the scale entry at s_tail (R2)
  atag_t       tgm [TD];
  logic [TDW-1:0] tg_t, tg_h;
  always_ff @(posedge clk) if (a_req) tgm[tg_t] <= a_tag;
  logic        rv1;
  logic [255:0] ln1;                         // the request's line
  logic [AB*8-1:0] gb1;                      // (AB > 32) the beat
  atag_t       tg1;
  function automatic logic [255:0] aline(input logic [AB*8-1:0] r, input logic [1:0] lx);
    aline = r[255:0];
    for (int l = 1; l < AB / 32; l++) if (2'(l) == lx) aline = r[256 * l +: 256];
  endfunction
  always_ff @(posedge clk) begin
    rv1 <= !rst && a_rvalid;
    if (a_rvalid) begin
      ln1 <= aline(a_rdata, tgm[tg_h].lx);
      gb1 <= a_rdata;
      tg1 <= tgm[tg_h];
    end
  end
  // the entry's data from the line: {Sw, the low block's}, {the high block's}
  logic [31:0] r_sw, r_swe, r_swr;          // the current rows' Sw; the line's words
  logic [95:0] r_wd0;
  logic [63:0] r_wd1;
  logic        r_we0, r_we1;
  always_comb begin
    logic [7:0]   xb;
    logic [31:0]  xw;
    logic [63:0]  x8;
    logic [127:0] x16;
    logic [63:0]  h0, h1;
    xb  = ln1[8 * tg1.doff +: 8];
    xw  = ln1[32 * tg1.doff[4:2] +: 32];
    x8  = ln1[64 * tg1.doff[4:3] +: 64];
    x16 = ln1[128 * tg1.doff[4] +: 128];
    if (tg1.rsc) begin
      h0 = '0; h1 = '0;
    end else if (tg1.wf == WF_W8) begin
      h0 = {56'd0, xb}; h1 = '0;
    end else if (tg1.wf == WF_NV) begin
      h0 = tg1.pair ? x16[63:0] : x8; h1 = tg1.pair ? x16[127:64] : x8;
    end else begin
      h0 = {32'd0, tg1.pair ? x8[31:0] : xw}; h1 = {32'd0, tg1.pair ? x8[63:32] : xw};
    end
    r_swe = ln1[32 * tg1.swo +: 32];
    r_swr = ln1[32 * tg1.sro +: 32];
    r_wd0 = {tg1.swe ? r_swe : r_sw, h0};
    r_wd1 = h1;
    r_we0 = rv1 && tg1.ent && tg1.hv != 2'b10 && !GA;
    r_we1 = rv1 && tg1.ent && tg1.hv[1] && !GA;
  end
  wire r_cpl = rv1 && tg1.ent && tg1.cpl;
  always_ff @(posedge clk)
    if (rv1 && !GA) begin
      if (tg1.swr) r_sw <= r_swr;
      else if (tg1.swe) r_sw <= r_swe;
    end
  logic [95:0] sr0;                          // the S0 scale entry: {Sw, low block}, high block
  logic [63:0] sr1;
  otpu_ram_1r1w #(.W(96), .N(DEPTH), .STYLE("block")) u_fs0 (
    .clk, .we(r_we0), .wa(s_tail), .wd(r_wd0), .re(1'b1), .ra(s_rd), .rd(sr0));
  otpu_ram_1r1w #(.W(64), .N(DEPTH), .STYLE("block")) u_fs1 (
    .clk, .we(r_we1), .wa(s_tail), .wd(r_wd1), .re(1'b1), .ra(s_rd), .rd(sr1));
  // GRP's entries (WG > 1), in the same slots: per chunk the group's Sw words (rows 0..7 and
  // 8..15: the Sw lines sl and sl + 1 of the beat that brought them, held in r_swv; RSC: the
  // entry's own beat) and the block's quarter lines, the beat's lines lx .. lx + 3 (int8:
  // quarter 0's low 8 bytes). An entry and a group's Sw words are aligned to their size (ssa
  // and srs are multiples of SWP, docs/isa.md "Weight groups"; AB 128): 4-bit's two lines at lx
  // 0 or 2, NVFP4's four at 0, the Sw lines at sl 0 or 2 (int8's one line, sl, anywhere), so
  // the lines past the first need no rotation of the beat
  logic [511:0]      gsw0;                  // the S0 entry's
  logic [3:0][255:0] gq0;
  if (WG > 1) begin : g_gsc
    logic [511:0] r_swv;
    logic [3:0][255:0] gl;
    wire gwe = rv1 && tg1.ent;
    // the Sw lines of an entry that brings them (swe: lines sl, sl + 1 of its beat; a Sw
    // request's are its beat's too), kept for the group's later entries
    wire [511:0] swl = {aline(gb1, tg1.sl | 2'd1), aline(gb1, tg1.sl)};
    always_ff @(posedge clk)
      if (rv1 && (tg1.swr || (tg1.ent && tg1.swe))) r_swv <= swl;
    always_comb begin
      gl[0] = (tg1.wf == WF_W8) ? {192'd0, ln1[64 * tg1.doff[4:3] +: 64]} : ln1;
      gl[1] = gb1[256 * (32'(tg1.lx) | 1) +: 256];
      gl[2] = gb1[512 +: 256];
      gl[3] = gb1[768 +: 256];
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && rv1 && tg1.wf != WF_W8 &&
          ((tg1.ent && !tg1.rsc && (tg1.lx & ((tg1.wf == WF_NV) ? 2'd3 : 2'd1)) != 2'd0) ||
           ((tg1.swr || (tg1.ent && tg1.swe)) && tg1.sl[0])))
        $fatal(1, "otpu_mxu: a group's scale lines off their size's alignment (lx %0d, sl %0d)",
               tg1.lx, tg1.sl);
`endif
    otpu_ram_1r1w #(.W(512), .N(DEPTH), .STYLE("block")) u_gsw (
      .clk, .we(gwe), .wa(s_tail), .wd(tg1.swe ? swl : r_swv), .re(1'b1), .ra(s_rd), .rd(gsw0));
    for (genvar q = 0; q < 4; q++) begin : g_q
      otpu_ram_1r1w #(.W(256), .N(DEPTH), .STYLE("block")) u_gq (
        .clk, .we(gwe && tg1.wq[q]), .wa(s_tail), .wd(gl[q]), .re(1'b1), .ra(s_rd),
        .rd(gq0[q]));
    end
  end else begin : g_ngs
    assign gsw0 = '0;
    assign gq0 = '0;
  end

  // ================================================================== compute pipeline
  typedef struct packed {
    logic       v;
    logic       last;     // last block of its row (the row's group)
    logic       first;    // first block of its row (the row's group)
    logic       h;        // 4-bit: the chunk's high half
    logic       o;        // PAIR: the chunk's high block is in the row (not past an odd KB)
  } cm_t;

  cm_t                 m0;
  logic [D*8-1:0]      w0;                  // the chunk (FIFO read register; WG > 1: lane 0's)
  logic [MCOLS-1:0][D*8-1:0] w0c;           // WG > 1: each column's part (lane j's)
  logic [MCOLS*D*8-1:0] a0;
  logic [MCOLS*64-1:0] asc0;
  logic [63:0] ascm;                        // GRP: the advance's row's scale record
  // S0's ACT RAM block and scale records are the ACT RAM's registered read; GRP (WG > 1): every
  // column's are the advance's row's, in its bank b0: column 0 takes them (the others the block
  // from their neighbour, a cycle later: the activation hop, g_ask; the record as ascm)
  logic pg0;
  logic [LCW-1:0] b0;
  logic kh0;                                // KT: S0's advance is an odd group's first (T_p)
  logic sl0;                                // SPL: S0's advance is a lo term (past the first)
  logic x10;                                // PU: S0's pass has a row 1
  logic [D*8-1:0] a1;                       // PU: row 1's block (bank b0 + 1), as a0's
  logic [63:0] ascm1;                       // PU: row 1's scale record
  if (WG > 1) begin : g_bc
    logic [D*8-1:0] ab;                     // column 0's block from bank b0 (before KT's swap)
    logic [D-1:0]   khs;                    // KT: byte i's swap (kh0 at its stage's read)
    logic [63:0]    ar;                     // the row's record (before KT's swap)
    if (AA) begin : g_ab
      // AASK: the ACT RAM reads byte i's stage i % CLS that many cycles after the pop (otpu_actram
      // ASK), its bank b0 as many cycles late (b0d[s - 1]), so a0's byte i is the skew's output
      (* shreg_extract = "no" *) logic [CLS-2:0][LCW-1:0] b0d;
      always_ff @(posedge clk) begin
        b0d[0] <= b0;
        for (int s = 1; s < CLS - 1; s++) b0d[s] <= b0d[s-1];
      end
      for (genvar i = 0; i < D; i++) begin : g_i
        wire [LCW-1:0] bs = (i % CLS == 0) ? b0 : b0d[(i % CLS == 0) ? 0 : i % CLS - 1];
        assign ab[i*8 +: 8] = act_data[(32'(bs)*D + i)*8 +: 8];
      end
      if (KTE) begin : g_khd                // (kh0 down b0d's line)
        (* shreg_extract = "no" *) logic [CLS-2:0] khd;
        always_ff @(posedge clk) begin
          khd[0] <= kh0;
          for (int s = 1; s < CLS - 1; s++) khd[s] <= khd[s-1];
        end
        for (genvar i = 0; i < D; i++) begin : g_i
          assign khs[i] = (i % CLS == 0) ? kh0 : khd[(i % CLS == 0) ? 0 : i % CLS - 1];
        end
      end else begin : g_nkhd
        assign khs = '0;
      end
    end else begin : g_nab
      assign ab = act_data[32'(b0)*D*8 +: D*8];
      assign khs = {D{kh0}};
    end
    assign ar = act_scale[32'(b0)*64 +: 64];
    // PU: row 1 from the odd bank beside b0 (no AASK: both rows' blocks are S0's reads), KT's
    // swap as row 0's
    if (PU) begin : g_r1
      wire [LCW-1:0] b1 = b0 | LCW'(1);
      wire [D*8-1:0] ab1 = act_data[32'(b1)*D*8 +: D*8];
      wire [63:0]    ar1 = act_scale[32'(b1)*64 +: 64];
      for (genvar i = 0; i < D; i++) begin : g_i
        assign a1[i*8 +: 8] = (KTE && kh0) ? ab1[(i ^ (D / 2))*8 +: 8] : ab1[i*8 +: 8];
      end
      assign ascm1 = (KTE && kh0) ? {ar1[47:32], ar1[63:48], ar1[31:0]} : ar1;
    end else begin : g_nr1
      assign a1 = '0;
      assign ascm1 = '0;
    end
    // KT: an odd group's first advance takes the tail block with its halves swapped, byte i from
    // byte i ^ D/2 (the same stage: D/2 is a multiple of CLS), one 2:1 after the bank's, and the
    // record's offsets da[b] from da[b ^ 2] ({da3, da2, da1, da0, Sa} -> {da1, da0, da3, da2, Sa})
    if (KTE) begin : g_kt
      for (genvar i = 0; i < D; i++) begin : g_i
        assign a0[i*8 +: 8] = khs[i] ? ab[(i ^ (D / 2))*8 +: 8] : ab[i*8 +: 8];
      end
      assign ascm = kh0 ? {ar[47:32], ar[63:48], ar[31:0]} : ar;
    end else begin : g_nkt
      assign a0[0 +: D*8] = ab;
      assign ascm = ar;
    end
    assign a0[D*8 +: (MCOLS-1)*D*8] = '0;           // (the columns past 0 hop: g_ask)
  end else begin : g_nbc
    assign a0 = act_data;
    assign ascm = '0;
    assign a1 = '0;
    assign ascm1 = '0;
  end
  assign asc0 = act_scale;
  logic cu0, pr0, rs0;
  logic [1:0] wf0;
  logic [MCOLS-1:0] hi0;                    // PAIR: the columns of the high block
  logic [7:0] M0;                           // the command's M (constant while its blocks flow)
  wire  w40 = (wf0 != WF_W8);
  // 4-bit elements: nibble i of the chunk half; int4 two's complement, E2M1 as twice its value
  // ({0, 1, 2, 3, 4, 6, 8, 12}, sign in bit 3); NVFP4 elements are E2M1's
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
    return dec4(w[(hb ? D*4 : 0) + 4*i +: 4], wf != 2'd1);
  endfunction
  // (WG > 1) nibble i of the chunk half hb, undecoded; a skewed lane's byte decoded: {high, low}
  // (int8 w = 16 h + l: the signed high nibble h, the unsigned low l, the packed PE's two
  // weights, joined at the column end: each output bit a function of one nibble and the format)
  function automatic logic [3:0] wnib(input logic [D*8-1:0] w, input int i, input logic hb);
    return w[(hb ? D*4 : 0) + 4*i +: 4];
  endfunction
  function automatic logic [15:0] wdec(input logic [7:0] b, input logic [1:0] wf);
    if (wf == WF_W8) return {{4{b[7]}}, b[7:4], 4'd0, b[3:0]};
    return {dec4(b[7:4], wf != 2'd1), dec4(b[3:0], wf != 2'd1)};
  endfunction

  // FIFO writes (the slot was reserved when the chunk was issued; see occ); the RAMs are
  // technology RAMs (rtl/tech)
  // (WG > 1: lane c's RAMs (g_fd lane 0's, g_fdw the others') hold every entry's lane c, written
  // at the lane's own tail by port W's lane c (w_bv[c]; the lanes arrive independently, each in
  // its requests' order). An entry is in the FIFO (f_cpl into f_count) once every lane has
  // written it; column j's part is lane j's (g_fdw). The chunk is stored here only: the adapter
  // passes the beats on as they return. XPOST: g_xp's lanes, written by the channels.)
  wire f_we = GA ? w_bv[0] : b_rvalid;                 // lane 0's write
  logic f_cpl;                                         // an entry completes (f_count)
  for (genvar g = 0; g < (XPOST ? 0 : FG); g++) begin : g_fd
    localparam int LO = fg_lo(g);
    localparam int W = ((g == FG - 1) ? FW : fg_lo(g + 1)) - LO;
    (* keep_hierarchy = "yes" *)
    otpu_ram_1r1w #(.W(W), .N(DEPTH), .STYLE(FU ? "ultra" : "block")) u_fd (
      .clk, .we(f_we), .wa(f_tl[g]), .wd(GA ? w_bd[LO +: W] : b_rdata[LO +: W]), .re(1'b1),
      .ra(f_rdl[g]), .rd(w0[LO +: W]));
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
  if (WG > 1 && !XPOST) begin : g_fdw
    // the other lanes, each with its own copies of its tail and of the read address (tl, rdl);
    // lc: per lane, the entries it has written past the complete ones (port W's lanes run ahead
    // of each other by at most the entries reserved: occ - f_count)
    (* keep *) logic [MCOLS-1:1][FG-1:0][PW-1:0] tl, rdl;
    logic [MCOLS-1:0][D*8-1:0] wl;                    // the lanes' read registers
    logic [MCOLS-1:0][PW:0] lc;
    always_comb begin
      logic c;
      c = 1'b1;                                        // the next entry is in every lane
      for (int l = 0; l < MCOLS; l++) c &= lc[l] != 0 || w_bv[l];
      f_cpl = c;
    end
    always_ff @(posedge clk)
      if (rst) begin
        tl <= '0; rdl <= '0; lc <= '0;
      end else begin
        for (int c = 1; c < MCOLS; c++)
          if (w_bv[c]) for (int g = 0; g < FG; g++) tl[c][g] <= tl[c][g] + 1'b1;
        for (int c = 0; c < MCOLS; c++)
          lc[c] <= lc[c] + (w_bv[c] ? 1'b1 : 1'b0) - (f_cpl ? 1'b1 : 1'b0);
        if (pop) begin
          if (last_k && !last_g) rdl <= {(MCOLS - 1) * FG{f_head}};
          else if (cdone && !ktt)                      // (KT: T_p stays the head)
            for (int c = 1; c < MCOLS; c++) for (int g = 0; g < FG; g++)
              rdl[c][g] <= rdl[c][g] + 1'b1;
        end
      end
    assign wl[0] = w0;
    for (genvar c = 1; c < MCOLS; c++) begin : g_c
      for (genvar g = 0; g < FG; g++) begin : g_fd
        localparam int LO = fg_lo(g);
        localparam int W = ((g == FG - 1) ? FW : fg_lo(g + 1)) - LO;
        (* keep_hierarchy = "yes" *)
        otpu_ram_1r1w #(.W(W), .N(DEPTH), .STYLE(FU ? "ultra" : "block")) u_fd (
          .clk, .we(w_bv[c]), .wa(tl[c][g]), .wd(w_bd[c*D*8 + LO +: W]), .re(1'b1),
          .ra(rdl[c][g]), .rd(wl[c][LO +: W]));
      end
    end
    assign w0c = wl;
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && fd_rst) begin
        for (int c = 0; c < MCOLS; c++) begin
          logic [PW-1:0] t;
          t = (c == 0) ? f_tail : tl[c][0];
          if (PW'(t - f_head - f_count) != PW'(lc[c]) || 32'(lc[c]) > 32'(occ) - 32'(f_count))
            $fatal(1, "otpu_mxu: FIFO lane %0d tail %0d, %0d entries ahead (head %0d, %0d in, %0d reserved)",
                   c, t, lc[c], f_head, f_count, occ);
          for (int g = 0; g < FG; g++)
            if (c != 0 && (tl[c][g] != tl[c][0] || rdl[c][g] != f_rd))
              $fatal(1, "otpu_mxu: FIFO lane %0d group %0d addresses %0d / %0d, not %0d / %0d", c,
                     g, tl[c][g], rdl[c][g], tl[c][0], f_rd);
        end
        if (b_rvalid) $fatal(1, "otpu_mxu: a port B chunk at WG > 1 (every MM in groups)");
      end
`endif
  end else if (XPOST) begin : g_xp
    // Lane c's RAMs, written by channel c on p_clk at the entry its request reserved, read at the
    // lane's copies of f_rd (rdl; lane 0 f_rdl). The entries complete in order: the next one
    // (f_tail) once each lane has a beat in its RAMs past lk, the lane's beats of the complete
    // entries (lw: p_cnt, the beats the lane's channel wrote; g_u: the beats it copied in).
    // g_a (option A, block RAM): the lane's RAMs one a quarter of the beat (QW bits), quarter q's
    // written by p_q's bit q (QB 0: all four by p_we, the remainder RAM by p_we with p_q its field
    // enables). A quarter's RAM holds its low QW - RB bits (252: 3.5 block RAMs of 512 x 72), and
    // the lane's remainder RAM (one RAMB18, 512 x 36 in four 9-bit fields) each quarter's top RB
    // in field g, its own field write; read at group 0's address (14.5 RAMB36 a lane for 16).
    localparam int QW = D * 2;
    localparam int RB = 4;
    (* keep *) logic [MCOLS-1:1][FG-1:0][PW-1:0] rdl;
    logic [MCOLS-1:0][D*8-1:0] wl;                    // the lanes' read registers
    logic [MCOLS-1:0][PW+1:0] lk, lw;
    always_comb begin
      f_cpl = 1'b1;
      for (int c = 0; c < MCOLS; c++) if (lw[c] == lk[c]) f_cpl = 1'b0;
    end
    always_ff @(posedge clk)
      if (rst) begin
        // (the counts do not restart with the MXU: the beats of its last requests are in)
        lk <= p_cnt; rdl <= '0;
      end else begin
        if (f_cpl) for (int c = 0; c < MCOLS; c++) lk[c] <= lk[c] + 1'b1;
        if (pop) begin
          if (last_k && !last_g) rdl <= {(MCOLS - 1) * FG{f_head}};
          else if (cdone && !ktt)                      // (KT: T_p stays the head)
            for (int c = 1; c < MCOLS; c++) for (int g = 0; g < FG; g++)
              rdl[c][g] <= rdl[c][g] + 1'b1;
        end
      end
    if (FU) begin : g_u
      // UltraRAM lanes (FIFO_STYLE "ultra"): UltraRAM has one clock, so the channel's writes land
      // first in a staging RAM of NS entries a lane (LUT RAM on two clocks: four quarter RAMs of
      // QW bits written as option A's, at the entry mod NS), and the copy moves each beat on clk
      // into the lane's UltraRAM (FG groups of 72-bit columns, whole beats) once p_cnt counts it.
      // A lane's beats arrive in its entries' order (its requests'), so the copy takes them in
      // turn: entry cs from staging entry cs mod NS (tk), its read registered (sd), then written
      // at s1 (v1), counted in lu. It takes one a cycle and a channel brings one every 4 aclk
      // cycles at the most (QB 2; QB 0, the simulation's, one a clk cycle), so NS covers p_cnt's
      // crossing; the simulation checks no post overruns the copy.
      localparam int NS = 32;
      localparam int NW = $clog2(NS);
      logic [MCOLS-1:0][PW+1:0] cc, lu;                // beats taken from staging, landed
      (* keep *) logic [MCOLS-1:0][3:0][PW-1:0] cs;    // the next entry (a copy a quarter RAM)
      (* keep *) logic [MCOLS-1:0][FG-1:0][PW-1:0] s1; // the landing entry (a copy a group)
      logic [MCOLS-1:0] tk, v1;
      always_comb for (int c = 0; c < MCOLS; c++) tk[c] = p_cnt[c] != cc[c];
      assign lw = lu;
      always_ff @(posedge clk) begin
        for (int c = 0; c < MCOLS; c++) s1[c] <= {FG{cs[c][0]}};
        if (rst) begin
          cc <= p_cnt; lu <= p_cnt; cs <= '0; v1 <= '0;
        end else begin
          v1 <= tk;
          for (int c = 0; c < MCOLS; c++) begin
            if (tk[c]) begin
              cc[c] <= cc[c] + 1'b1;
              for (int q = 0; q < 4; q++) cs[c][q] <= cs[c][q] + 1'b1;
            end
            if (v1[c]) lu[c] <= lu[c] + 1'b1;
          end
        end
      end
      for (genvar c = 0; c < MCOLS; c++) begin : g_c
        logic [D*8-1:0] sd;                            // the staged beat (its read register)
        logic [3:0] swe;
        logic [3:0][NW-1:0] swa, sra;
        logic [3:0][QW-1:0] swd;
        for (genvar q = 0; q < 4; q++) begin : g_s
          if (QB == 0) begin : g_b
            assign swd[q] = p_d[c][QW*q +: QW];
          end else begin : g_q
            assign swd[q] = p_d[c][QW-1:0];
          end
          assign swe[q] = (QB == 0) ? p_we[c] : p_q[c][0][(QB > 0) ? q : 0];
          assign swa[q] = p_slot[c][q][NW-1:0];
          assign sra[q] = cs[c][q][NW-1:0];
        end
        (* keep_hierarchy = "yes" *)
        otpu_mxu_stage #(.QW(QW), .N(NS)) u_sg (.wclk(p_clk), .we(swe), .wa(swa), .wd(swd),
          .rclk(clk), .re(tk[c]), .ra(sra), .rd(sd));
        if (WA) begin : g_wa
          // WASK: the lane's FC columns (SPC bytes each, stage-major: wa_byte), written at the
          // landing as the groups' were, column m read at ar, the lane's read address delayed
          // to when its PEs take its first stage wa_sm(m): the byte of stage s at lane c is the
          // PE's c + s + 1 edges after the pop (the skews', g_wcol), through the UltraRAM's output
          // register (OREG) and ar's tap c + wa_sm(m) - 1, or (lane 0's first column) no OREG at
          // tap 0; the column's bytes of the next stage one register later (u_e). PU: stage s's
          // byte at c + wa_t(s) + 1 edges (otpu_mxpc's weights, ceil(s / 2) + c after S0)
          // ar's taps are data, outside the reset, as the skews' bytes are without WA (g_wcol's
          // g_s): each column reads at its tap every cycle, and the PEs take its bytes only at a
          // pop's delays, where the tap holds that pop's address (f_rdl / rdl, reset). With a
          // reset, a BCU1525 build at 150 MHz (1,540 of these flip-flops, unpumped) had its worst
          // core paths from rst_mxu's copies into their R pins: no logic, 5.7 ns of route
          localparam int AD = wa_t(wa_sm(FC - 1)) + c - 1;     // the longest tap
          (* shreg_extract = "no" *) logic [AD:0][PW-1:0] ar;
          assign ar[0] = (c == 0) ? f_rdl[0] : rdl[(c == 0) ? 1 : c][0];
          always_ff @(posedge clk)
            for (int k = 1; k <= AD; k++) ar[k] <= ar[k-1];
          for (genvar m = 0; m < FC; m++) begin : g_m
            localparam int SM = wa_t(wa_sm(m));
            localparam bit OR = SM + c > 0;
            localparam int DL = OR ? SM + c - 1 : 0;
            logic [FB-1:0] wdm, rdm;
            for (genvar r = 0; r < SPC; r++) begin : g_r
              localparam int K = SPC * m + r;
              if (K < D) begin : g_b
                assign wdm[8*r +: 8] = sd[8*wa_byte(K) +: 8];
                otpu_skew #(.W(8), .N(wa_t(K / BPS) - SM)) u_e (.clk, .en(1'b1),
                                                                .d(rdm[8*r +: 8]),
                                                                .q(wsk[c][wa_byte(K)]));
              end else begin : g_z
                assign wdm[8*r +: 8] = '0;
              end
            end
            (* keep_hierarchy = "yes" *)
            otpu_ram_1r1w #(.W(FB), .N(DEPTH), .OREG(OR), .STYLE("ultra")) u_fd (
              .clk, .we(v1[c]), .wa(s1[c][m % FG]), .wd(wdm), .re(1'b1), .ra(ar[DL]), .rd(rdm));
          end
          assign wl[c] = '0;
`ifndef SYNTHESIS
          // no write lands in an entry a column still reads (av: a pop k cycles ago, its entry
          // at ar[k]): the issuer reuses no entry freed in the last AW cycles (occz)
          logic [AD:0] av;
          assign av[0] = pop;
          always @(posedge clk) for (int k = 1; k <= AD; k++) av[k] <= !rst && av[k-1];
          always @(posedge clk)
            if (!rst && fd_rst && v1[c])
              for (int k = 0; k <= AD; k++)
                if (av[k] && ar[k] == s1[c][0])
                  $fatal(1, "otpu_mxu: lane %0d lands entry %0d, read %0d cycles after its pop",
                         c, s1[c][0], k);
`endif
        end else begin : g_nwa
          for (genvar g = 0; g < FG; g++) begin : g_fd
            localparam int LO = fg_lo(g);
            localparam int W = ((g == FG - 1) ? FW : fg_lo(g + 1)) - LO;
            (* keep_hierarchy = "yes" *)
            otpu_ram_1r1w #(.W(W), .N(DEPTH), .STYLE("ultra")) u_fd (
              .clk, .we(v1[c]), .wa(s1[c][g]), .wd(sd[LO +: W]), .re(1'b1),
              .ra((c == 0) ? f_rdl[g] : rdl[(c == 0) ? 1 : c][g]), .rd(wl[c][LO +: W]));
          end
        end
      end
`ifndef SYNTHESIS
      always @(posedge p_clk)
        if (!rst && fd_rst)
          for (int c = 0; c < MCOLS; c++)
            if (p_we[c] && 32'(PW'(p_slot[c][0] - cs[c][0])) >= NS)
              $fatal(1, "otpu_mxu: lane %0d posts entry %0d, %0d past the copy's %0d (staging %0d)",
                     c, p_slot[c][0], PW'(p_slot[c][0] - cs[c][0]), cs[c][0], NS);
      always @(posedge clk)
        if (!rst && fd_rst)
          for (int c = 0; c < MCOLS; c++)
            for (int q = 1; q < 4; q++)
              if (cs[c][q] != cs[c][0] || s1[c][q % FG] != s1[c][0])
                $fatal(1, "otpu_mxu: lane %0d's copy addresses differ", c);
      // an entry completes (f_cpl, at f_tail) only once every lane's beat has landed in its
      // UltraRAM, at an edge before: lnd, the entries each lane's RAMs hold, set by the writes
      // (v1 at s1) after this edge's completion is checked, cleared as the entry completes
      bit [DEPTH-1:0] lnd [MCOLS];
      always @(posedge clk)
        if (rst) begin
          for (int c = 0; c < MCOLS; c++) lnd[c] = '0;
        end else begin
          if (f_cpl && fd_rst)
            for (int c = 0; c < MCOLS; c++) begin
              if (!lnd[c][f_tail])
                $fatal(1, "otpu_mxu: entry %0d complete before lane %0d's beat landed (UltraRAM)",
                       f_tail, c);
              lnd[c][f_tail] = 1'b0;
            end
          for (int c = 0; c < MCOLS; c++) if (v1[c]) lnd[c][s1[c][0]] = 1'b1;
        end
`endif
    end else begin : g_a
      initial if (FG != 4) $fatal(1, "otpu_mxu: XPOST: FG %0d, not 4", FG);
      assign lw = p_cnt;
      for (genvar c = 0; c < MCOLS; c++) begin : g_c
        logic [FG-1:0]     rbe;                      // the remainder's field writes
        logic [FG*9-1:0]   rwd, rrd;
        for (genvar g = 0; g < FG; g++) begin : g_fd
          logic [QW-1:0] wd;
          if (QB == 0) begin : g_b
            assign wd = p_d[c][QW*g +: QW];
          end else begin : g_q
            assign wd = p_d[c][QW-1:0];
          end
          assign rbe[g] = QB == 0 || p_q[c][1][(QB > 0) ? g : 0];
          assign rwd[9*g +: 9] = 9'(wd[QW-1 -: RB]);
          assign wl[c][QW*g + QW - RB +: RB] = rrd[9*g +: RB];
          (* keep_hierarchy = "yes" *)
          otpu_ram_1r1w_2c #(.W(QW - RB), .N(DEPTH), .STYLE("block")) u_fd (
            .wclk(p_clk), .we((QB == 0) ? p_we[c] : p_q[c][0][(QB > 0) ? g : 0]),
            .wa(p_slot[c][g]), .wd(wd[QW-RB-1:0]),
            .rclk(clk), .re(1'b1), .ra((c == 0) ? f_rdl[g] : rdl[(c == 0) ? 1 : c][g]),
            .rd(wl[c][QW*g +: QW - RB]));
        end
        (* keep_hierarchy = "yes" *)
        otpu_ram_1r1w_2c_be #(.W(FG * 9), .N(DEPTH), .NB(FG), .STYLE("block")) u_fr (
          .wclk(p_clk), .we(p_we[c]), .wbe(rbe), .wa(p_slot[c][4]), .wd(rwd), .rclk(clk),
          .re(1'b1), .ra((c == 0) ? f_rdl[0] : rdl[(c == 0) ? 1 : c][0]), .rd(rrd));
      end
    end
    assign w0 = wl[0];
    assign w0c = wl;
`ifndef SYNTHESIS
    // the posted beats: each into an entry reserved and not complete; each lane's beats past lk
    // within the entries reserved
    always @(posedge p_clk)
      if (!rst && fd_rst)
        for (int c = 0; c < MCOLS; c++)
          if (p_we[c] && 32'(PW'(p_slot[c][0] - f_tail)) >= 32'(occ) - 32'(f_count))
            $fatal(1, "otpu_mxu: lane %0d posts into entry %0d (in: to %0d; %0d reserved)",
                   c, p_slot[c][0], f_tail, 32'(occ) - 32'(f_count));
          else if (QB > 0 && (p_we[c] ? !$onehot(p_q[c][0]) : p_q[c][0] != '0))
            $fatal(1, "otpu_mxu: lane %0d's p_q %b with p_we %0d (one-hot with it, else 0)",
                   c, p_q[c][0], p_we[c]);
          else if (p_we[c] && (p_slot[c] != {5{p_slot[c][0]}} || p_q[c][1] != p_q[c][0]))
            $fatal(1, "otpu_mxu: lane %0d's copies differ (p_slot %h, p_q %h)", c, p_slot[c],
                   p_q[c]);
    always @(posedge clk)
      if (!rst && fd_rst) begin
        for (int c = 0; c < MCOLS; c++)
          if (32'((PW+2)'(p_cnt[c] - lk[c])) > 32'(occ) - 32'(f_count))
            $fatal(1, "otpu_mxu: lane %0d counts %0d beats past its complete entries, %0d reserved",
                   c, (PW+2)'(p_cnt[c] - lk[c]), 32'(occ) - 32'(f_count));
        if (b_rvalid || w_bv != '0) $fatal(1, "otpu_mxu: XPOST: a chunk returned to the MXU");
        if (i_slot != PW'(f_head + occ))
          $fatal(1, "otpu_mxu: request entry %0d, not head %0d + %0d", i_slot, f_head, occ);
      end
`endif
  end else begin : g_nfdw
    assign f_cpl = f_we;
    for (genvar c = 0; c < MCOLS; c++) begin : g_c
      assign w0c[c] = w0;
    end
  end
  // S0: the popped chunk, its ACT RAM block and records, its scale entry
  always_ff @(posedge clk) begin
    pg0 <= c_pg;
    b0 <= act_row[LCW-1:0];
    kh0 <= c_kt && c_odd && ck == 16'd0;
    sl0 <= SPE && pop_e.spl && ck != 16'd0;
    x10 <= c_x1;
    m0 <= '0;
    if (pop) begin
      m0.v <= 1'b1;
      m0.last <= last_k;
      m0.first <= (ck == 16'd0);
      m0.h <= c_w4 && !c_1a && ck[0];
      m0.o <= c_pair && !(last_k && c_KB[0]);
    end
    cu0 <= c_unit;
    rs0 <= c_rsc;
    wf0 <= pwf;
    pr0 <= c_pair;
    hi0 <= c_hi;
    M0 <= p_M;
  end

  // ---- the shifts: per block its eight groups' W - dw and m (S1), per column and group
  // B = sh >= 0 ? m << sh : 0 with sh = W - dw - da (S2), skewed to the column end
  typedef struct packed {
    logic [7:0][10:0] s;       // W - dw, signed
    logic [7:0][5:0]  m;
  } bdec_t;
  // (lo: SPL's lo term, under RSC: W - 8)
  function automatic bdec_t bdec(input logic [63:0] b, input logic [1:0] wf, input logic u,
                                 input logic lo);
    bdec_t r;
    logic [10:0] w;
    w = (wf == WF_W8) ? 11'd16 : (wf == WF_NV) ? 11'd13 : 11'd11;
    for (int g = 0; g < 8; g++) begin
      logic [3:0] ex;
      ex = b[8*g + 3 +: 4];
      if (u) begin
        r.s[g] = lo ? w - 11'd8 : w; r.m[g] = 6'd1;
      end else if (wf == WF_W8) begin
        r.s[g] = w - 11'(b[7:0]); r.m[g] = 6'd1;
      end else if (wf == WF_NV) begin
        // E4M3 s = {0, E, M}: m = E ? 8 + M : M, dw = 15 - max(E, 1): W - dw = max(E, 1) - 2
        r.m[g] = (ex != 4'd0) ? {3'b001, b[8*g +: 3]} : {3'b000, b[8*g +: 3]};
        r.s[g] = 11'((ex == 4'd0) ? 4'd1 : ex) - 11'd2;
      end else begin
        r.s[g] = w - 11'(b[7:0]); r.m[g] = b[8 + 6*(g/2) +: 6];
      end
    end
    return r;
  endfunction
  // The format's copies (WG > 1): one a column and chain, each loaded with c_wf as wf0 is (kept
  // instances: identical registers merge), for the column's decodes, its shifts' (dcol1) and its
  // weights' at the skews' ends (wdec). wf0's copies into every PE's DSP inputs were a WG 8
  // slice's worst class at 166.67 MHz (wf0_reg_rep -> u_pe's A / D data, 88 of the worst 200)
  localparam int NCG = D / CLS;             // chains a column
  logic [1:0] wfc [MCOLS][NCG];
  for (genvar j = 0; j < MCOLS; j++) begin : g_wfc
    for (genvar g = 0; g < NCG; g++) begin : g_g
      if (WG > 1) begin : g_c
        (* keep_hierarchy = "yes" *)
        otpu_delay #(.W(2), .N(1)) u_wf (.clk, .en(1'b1), .d(c_wf), .q(wfc[j][g]));
      end else begin : g_n
        assign wfc[j][g] = wf0;
      end
    end
  end
  // The decoded blocks (S1), a registered copy per column (g_bd: kept instances, since identical
  // registers merge) beside the column's shifts: one register driving every column's was the
  // ASIC card's worst path (r4: a DFF_X1 of dhi1 into the four columns' bm2 across the die,
  // 1.01 ns of clock-to-Q into its load). WG 1: the advance's (natural) blocks, the low one the
  // advance's half (4-bit without PAIR) or the entry's low block, the high one PAIR's; WG > 1:
  // each column's own rows' (dcol1, below), the copies the advance's valid alone
  bdec_t dlo1 [MCOLS];
  bdec_t dhi1 [MCOLS];
  logic [MCOLS-1:0] vlo1, vhi1;
  logic [MCOLS-1:0][3:0][7:0] da1;
  logic [MCOLS-1:0] hi1;
  if (WG == 1) begin : g_nd
    wire [63:0] blo0 = (w40 && !pr0 && m0.h) ? sr1 : sr0[63:0];
    bdec_t dlo0, dhi0;
    always_comb begin
      dlo0 = bdec(blo0, wf0, cu0 || rs0, 1'b0);
      dhi0 = bdec(sr1, wf0, cu0 || rs0, 1'b0);
    end
    for (genvar j = 0; j < MCOLS; j++) begin : g_bd
      (* keep_hierarchy = "yes" *)
      otpu_delay #(.W(2 * $bits(bdec_t) + 2), .N(1)) u_bd (.clk, .en(1'b1),
        .d({dlo0, dhi0, m0.v, m0.v && m0.o}), .q({dlo1[j], dhi1[j], vlo1[j], vhi1[j]}));
    end
    always_ff @(posedge clk)
      for (int j = 0; j < MCOLS; j++) begin
        da1[j] <= asc0[j*64 + 32 +: 32];
        hi1[j] <= hi0[j];
      end
  end else begin : g_gd
    for (genvar j = 0; j < MCOLS; j++) begin : g_bd
      (* keep_hierarchy = "yes" *)
      otpu_delay #(.W(1), .N(1)) u_bd (.clk, .en(1'b1), .d(m0.v), .q(vlo1[j]));
      assign dlo1[j] = '0;
      assign dhi1[j] = '0;
    end
    assign vhi1 = '0;
    assign hi1 = '0;
    // (every column against the advance's row: its offsets)
    always_ff @(posedge clk) for (int j = 0; j < MCOLS; j++) da1[j] <= ascm[32 +: 32];
  end
  // GRP (WG > 1): each column's block data are its own rows', from the entry's quarters: int8
  // row j's byte; 4-bit (t: the column's first or second row, 2j + t) int4x / fp4x6 its word
  // (rows 8.. in quarter 1), NVFP4 its 8 bytes (rows 4r .. 4r + 3 in quarter r)
  function automatic logic [63:0] gent(input logic [3:0][255:0] q, input int j, input int t,
                                       input logic [1:0] wf);
    int r;
    r = 2 * j + t;
    if (wf == WF_W8) return {56'd0, q[0][8*j +: 8]};
    if (wf == WF_NV) return q[r / 4][64*(r % 4) +: 64];
    return {32'd0, q[r / 8][32*(r % 8) +: 32]};
  endfunction
  bdec_t dcol1 [MCOLS], dcol1b [MCOLS];
  logic  pk1, w81;
  if (WG > 1) begin : g_gdec
    always_ff @(posedge clk) begin
      pk1 <= pg0;
      w81 <= (wf0 == WF_W8);
      for (int j = 0; j < MCOLS; j++) begin
        dcol1[j] <= bdec(gent(gq0, j, 0, wfc[j][0]), wfc[j][0], cu0 || rs0, sl0);
        dcol1b[j] <= bdec(gent(gq0, j, 1, wfc[j][0]), wfc[j][0], cu0 || rs0, sl0);
      end
    end
  end else begin : g_ngdec
    assign pk1 = 1'b0;
    assign w81 = 1'b0;
    for (genvar j = 0; j < MCOLS; j++) begin : g_j
      assign dcol1[j] = '0;
      assign dcol1b[j] = '0;
    end
  end
  logic [MCOLS-1:0][7:0][16:0] bm2;
  always_ff @(posedge clk)
    for (int j = 0; j < MCOLS; j++)
      for (int g = 0; g < 8; g++) begin
        logic [10:0] sh;
        logic [10:0] s;
        logic [5:0]  m;
        logic        v;
        s = GA ? dcol1[j].s[g] : hi1[j] ? dhi1[j].s[g] : dlo1[j].s[g];
        m = GA ? dcol1[j].m[g] : hi1[j] ? dhi1[j].m[g] : dlo1[j].m[g];
        v = hi1[j] ? vhi1[j] : vlo1[j];
        sh = s - 11'(da1[j][g / 2]);
        bm2[j][g] <= (v && !sh[10]) ? 17'(17'(m) << sh[4:0]) : 17'd0;
      end
  // pg (WG > 1): the B of each column's second row, 2j + 1 (its packed sums' high halves), and
  // pg and int8 at S2
  logic [MCOLS-1:0][7:0][16:0] bm2b;
  logic pk2, w82;
  if (WG > 1) begin : g_bm2b
    always_ff @(posedge clk) begin
      pk2 <= pk1;
      w82 <= w81;
      for (int j = 0; j < MCOLS; j++)
        for (int g = 0; g < 8; g++) begin
          logic [10:0] sh;
          logic        v;
          v = pk1 && vlo1[j];
          sh = dcol1b[j].s[g] - 11'(da1[j][g / 2]);
          bm2b[j][g] <= (v && !sh[10]) ? 17'(17'(dcol1b[j].m[g]) << sh[4:0]) : 17'd0;
        end
    end
  end else begin : g_nbm2b
    assign bm2b = '0;
    assign pk2 = 1'b0;
    assign w82 = 1'b0;
  end
  // the row's Sw (UNIT: 1) and each column's Sa * 2^-W, with every advance (the row's last
  // carries them to the result FIFO): fp_mul's flush and rounding at the bottom of the range
  wire [31:0] sw0 = cu0 ? F_ONE : sr0[95:64];
  wire [31:0] wpow = (wf0 == WF_W8) ? 32'h3780_0000 : (wf0 == WF_NV) ? 32'h3900_0000
                                                                     : 32'h3A00_0000;
  // x * 2^-k as fp_mul computes it (flushed operands and result, the canonical NaN), without a
  // multiplier: the exponent less k; a result under the normal range flushes to a signed zero,
  // except the one that rounds up to the smallest normal (exponent k, fraction all ones)
  function automatic f32_t fp_pow2m(input f32_t x, input logic [4:0] k);
    logic [8:0] e;
    if (x[30:23] == 8'd0) return {x[31], 31'd0};
    if (x[30:23] == 8'hFF) return (x[22:0] != 0) ? F_NAN : x;
    e = 9'(x[30:23]) - 9'(k);
    if (!e[8] && e != 0) return {x[31], e[7:0], x[22:0]};
    if (e == 0 && x[22:0] == 23'h7F_FFFF) return {x[31], 8'd1, 23'd0};
    return {x[31], 31'd0};
  endfunction
  wire [4:0] wk = (wf0 == WF_W8) ? 5'd16 : (wf0 == WF_NV) ? 5'd13 : 5'd11;
  logic [31:0] sw1, sw1x;                   // (sw1x: PU's row 1's)
  logic [MCOLS-1:0][31:0] sap1, sap1b;
  // (GRP: the columns' factors are their rows' Sw, the row factor the activation row's Sa', which
  // the drain's lanes swap back: the products' order is the ISA's; pg: rows 8..15's in sap1b)
  always_ff @(posedge clk) begin
    sw1 <= GA ? fp_pow2m(ascm[31:0], wk) : sw0;
    sw1x <= PU ? fp_pow2m(ascm1[31:0], wk) : '0;
    for (int j = 0; j < MCOLS; j++) begin
      sap1[j] <= GA ? (cu0 ? F_ONE : gsw0[32*j +: 32])
                                  : fp_pow2m(asc0[j*64 +: 32], wk);
      sap1b[j] <= cu0 ? F_ONE : gsw0[256 + 32*j +: 32];
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    for (int j = 0; j < MCOLS; j++)
      if (!rst && rst_seen && fp_pow2m(act_scale[j*64 +: 32], wk) != fp_mul(act_scale[j*64 +: 32], wpow))
        $fatal(1, "otpu_mxu: Sa %h * 2^-%0d is not fp_mul's", act_scale[j*64 +: 32], wk);
`endif

  // ---- the systolic array (docs/mxu_systolic.md). The chunk is decoded once (S1), position i
  // is delayed by its chain stage k = i % CLS, and the weights then move one register hop per
  // column: column j sees them j cycles after column 0. Both streams flow: the low block (or the
  // 4-bit half of a non-PAIR advance) and PAIR's high block; each column takes one in its DSPs'
  // pre-adder (A = low, D = high, INMODE by its per-command hi0[j]). Column j's activation byte i
  // is delayed k + j (shift registers) into the DSP's B register. Along D: chains of CLS products,
  // one per DSP48 (otpu_pe: M register, P = PCIN + M); a chain's end (S0 + CLS + 2 + j, in the
  // column end's input registers) is one group's sum. Exact integers.
  // the weights into column j's PEs: the hop (WG 1), or (WG > 1) each column's own part, skewed
  // by j more, its PEs' weight register the last. At WG > 1 the PEs are packed ones (PK: sel 1,
  // the high weight the packed second row's): the lane's byte as it is (int8's, pg's rows 2j and
  // 2j + 1's elements)
  logic [D-1:0][15:0] wpe [MCOLS];
  if (WG > 1 && !PU) begin : g_wcol
    // A lane's weights are skewed as they arrive, a byte (int8's, or 4-bit's two nibbles: the
    // high weight's, the low's), and decoded at the skew's end into the PE's weight register
    // (columns past the first: the DSP's A / D registers; column 0: a register of its own, the
    // skew one stage shorter), by the column and chain's copy of the format (wfc). The decode
    // takes the pop head's format: the next command pops behind the head only at the same
    // format (the pn test), so a skew holds one format.
    // WA: the lane's bytes come at the skews' ends from the lane's columns (g_xp g_u's g_wa)
    for (genvar j = 0; j < MCOLS; j++) begin : g_j
      logic [D-1:0][7:0] wr;                         // the lane's byte (S0)
      logic [D-1:0][7:0] wrj;                        // ... registered (S1)
      always_comb for (int i = 0; i < D; i++) wr[i] = w0c[j][i*8 +: 8];
      always_ff @(posedge clk) wrj <= wr;
      for (genvar i = 0; i < D; i++) begin : g_i
        localparam int N = i % CLS + j - ((j != 0) ? 1 : 0);   // the decoded weights' stages
        if (WA && j != 0) begin : g_a
          assign wpe[j][i] = wdec(wsk[j][i], wfc[j][i / CLS]);
        end else if (WA) begin : g_a0
          logic [15:0] wq;
          always_ff @(posedge clk) wq <= wdec(wsk[j][i], wfc[j][i / CLS]);
          assign wpe[j][i] = wq;
        end else if (j != 0) begin : g_s
          logic [7:0] ws;
          otpu_skew #(.W(8), .N(N)) u_w (.clk, .en(1'b1), .d(wrj[i]), .q(ws));
          assign wpe[j][i] = wdec(ws, wfc[j][i / CLS]);
        end else if (N != 0) begin : g_s0
          logic [7:0]  ws;
          logic [15:0] wq;
          otpu_skew #(.W(8), .N(N - 1)) u_w (.clk, .en(1'b1), .d(wrj[i]), .q(ws));
          always_ff @(posedge clk) wq <= wdec(ws, wfc[j][i / CLS]);
          assign wpe[j][i] = wq;
        end else begin : g_d0
          logic [15:0] wq;
          always_ff @(posedge clk) wq <= wdec(wr[i], wfc[j][i / CLS]);
          assign wpe[j][i] = wq;
        end
      end
    end
  end else if (WG == 1) begin : g_whop               // (PU: g_pcol's columns decode their own)
    logic [D-1:0][15:0] wd;                           // S1: {high block, low block / half}
    always_ff @(posedge clk)
      for (int i = 0; i < D; i++) wd[i] <= {wsel(w0, i, 1'b1, wf0), wsel(w0, i, m0.h, wf0)};
    logic [D-1:0][15:0] wc [MCOLS];                  // the streams at column j
    for (genvar i = 0; i < D; i++) begin : g_wsk
      otpu_skew #(.W(16), .N(i % CLS)) u_w (.clk, .en(1'b1), .d(wd[i]), .q(wc[0][i]));
    end
    always_ff @(posedge clk)
      for (int j = 1; j < MCOLS; j++) wc[j] <= wc[j-1];
    for (genvar j = 0; j < MCOLS; j++) begin : g_j
      assign wpe[j] = wc[(j == 0) ? 0 : j - 1];
    end
  end
  logic [7:0] as_k [MCOLS][D];
  logic signed [PRW-1:0] preg [MCOLS][D];             // running sums (the DSPs' P registers;
                                                      // a chain's last: its P, unregistered)
  logic [47:0] pc [MCOLS][D];                         // their cascade outputs
`ifndef SYNTHESIS
  bit hi_v;                                           // hi0 and the PEs' selects are loaded
  initial hi_v = 1'b0;
  always @(posedge clk) hi_v <= 1'b1;
`endif
  // The activations: column j's block byte i through its own skew (i % CLS + j stages) or, at
  // WG > 1 (j > 0), column j - 1's a cycle later (every column takes the advance's row's block,
  // column 0 from its bank: g_bc): the activation hop. AASK: column 0's skew is the ACT RAM's
  // read by stage (g_bc g_ab), none here
  for (genvar j = 0; j < (PU ? 0 : MCOLS); j++) begin : g_ask     // (PU: g_pcol)
    for (genvar i = 0; i < D; i++) begin : g_p
      if (WG > 1 && j != 0) begin : g_hop
        logic [7:0] ar;                               // column j - 1's, a cycle later
        always_ff @(posedge clk) ar <= as_k[j-1][i];
        assign as_k[j][i] = ar;
      end else begin : g_own
        otpu_skew #(.W(8), .N(AA ? 0 : i % CLS + j)) u_a (.clk, .en(1'b1),
                                                        .d(a0[(j*D + i)*8 +: 8]), .q(as_k[j][i]));
      end
      // columns past the first register their weights in the DSP (WREG), from the previous
      // column's hop (a copy of wc[j]; WG > 1: the last stage of the column's own skew)
      otpu_pe #(.FIRST(i % CLS == 0), .LAST(i % CLS == CLS - 1), .WREG(j != 0), .PK(WG > 1)) u_pe (
        .clk, .en(1'b1), .act(as_k[j][i]), .wlo(wpe[j][i][7:0]), .whi(wpe[j][i][15:8]),
        .sel(GA || c_hi[j]),
        .pcin((i % CLS == 0) ? 48'd0 : pc[j][(i % CLS == 0) ? i : i - 1]), .pcout(pc[j][i]),
        .p(preg[j][i]));
`ifndef SYNTHESIS
      // the DSP's select register is hi0[j] (both are c_hi[j], loaded every cycle; WG > 1: 1)
      always @(posedge clk) begin
        if (hi_v && u_pe.selr != (GA || hi0[j]))
          $fatal(1, "otpu_mxu: PE %0d.%0d's select %0d is not hi0 %0d", j, i, u_pe.selr,
                 GA || hi0[j]);
      end
      if (WG == 1 && j != 0) begin : g_wck
        always @(posedge clk)
          if (hi_v && {u_pe.whr, u_pe.wlr} != g_whop.wc[j][i])
            $fatal(1, "otpu_mxu: PE %0d.%0d's weights are not wc", j, i);
      end
`endif
    end
  end

  // ---- column ends: group g's chain ends (CPG of them: the first on A, the second on D) with
  // their B (skewed from S2 to the column end's input registers, S0 + CLS + 2 + j), the row's
  // first block at the accumulator's OPMODE register (two cycles later), the row sum in the
  // accumulator CLS + 6 + j edges after S0, deskewed to S0 + LROW
  // WG > 1, the packed select (pg, int8): a column's chain ends are packed sums
  // P = S0 + 2^15 + S1 * 2^16 (|S0|, |S1| < 2^15; the chain's first PE adds the 2^15), so
  // S0 = P[15:0] - 2^15 (bit 15 inverted) and S1 = P >> 16, wires: the column end takes S0 (pkd:
  // packed at the chain end, skewed with B) and, int8 (w = 16 h + l), 16 S1 on its pre-adder's D
  // (w8d), the row's sum; pg a second one (u_ce2) S1 with the second row's B, acc2_al its sums
  logic [MCOLS-1:0][47:0] acc_al, acc2_al;
  // BRAW (BR, WG > 1): a column's B are its scale entry's rows' bytes (gent, S0) with the row's
  // offsets, the format, UNIT and the valid, skewed to the column end (166 bits a column, not B's
  // 2 x 136), decoded and shifted there in two registers (S1 and S2's)
  localparam bit BR = BRAW && GA;
  typedef struct packed {
    logic [63:0] r0, r1;                    // gent: the column's rows 2j (int8: j) and 2j + 1
    logic [31:0] da;                        // the advance's row's offsets
    logic [1:0]  wf;
    logic        u, v, pg, w8;              // UNIT / RSC, a block, pg, int8
    logic        lo;                        // SPL: a lo term
  } braw_t;
  typedef struct packed {                   // PU: and row 1's offsets
    braw_t       b;
    logic [31:0] da1;
  } praw_t;
  for (genvar j = 0; j < (PU ? 0 : MCOLS); j++) begin : g_col_end // (PU: g_pcol)
    logic [7:0][23:0] ca, cd;
    logic [7:0][16:0] bmd, bmdb;            // the B at the column end (bmdb: pg's second row's)
    logic             fst, pkd, w8d;
    logic [47:0]      acc;
    always_comb
      for (int g = 0; g < 8; g++) begin
        logic [PRW-1:0] pe;
        logic [31:0]    px;                         // (PRW 32 at WG > 1)
        pe = preg[j][g * CPG * CLS + CLS - 1];
        px = 32'(pe);
        ca[g] = (WG > 1 && pkd) ? {{9{!pe[15]}}, pe[14:0]} : pe[23:0];
        cd[g] = (WG > 1) ? {{4{px[31]}}, px[31:16], 4'd0}
              : (CPG == 2) ? 24'(preg[j][(g * CPG + 1) * CLS + CLS - 1]) : 24'd0;
      end
    if (BR) begin : g_br
      braw_t       rw0, rwk;
      bdec_t       e1, e1b;
      logic [31:0] da1k;
      logic        v1k, pk1k, w81k;
      assign rw0 = '{r0: gent(gq0, j, 0, wfc[j][0]), r1: gent(gq0, j, 1, wfc[j][0]),
                     da: ascm[32 +: 32], wf: wfc[j][0], u: cu0 || rs0, v: m0.v, pg: pg0,
                     w8: wf0 == WF_W8, lo: sl0};
      otpu_skew #(.W($bits(braw_t)), .N(CLS - 1 + j)) u_bm (.clk, .en(1'b1), .d(rw0), .q(rwk));
      always_ff @(posedge clk) begin
        e1 <= bdec(rwk.r0, rwk.wf, rwk.u, rwk.lo);
        e1b <= bdec(rwk.r1, rwk.wf, rwk.u, rwk.lo);
        da1k <= rwk.da;
        v1k <= rwk.v;
        pk1k <= rwk.pg;
        w81k <= rwk.w8;
        pkd <= pk1k || w81k;
        w8d <= w81k;
        for (int g = 0; g < 8; g++) begin
          logic [10:0] sh, shb;
          sh = e1.s[g] - 11'(da1k[8 * (g / 2) +: 8]);
          shb = e1b.s[g] - 11'(da1k[8 * (g / 2) +: 8]);
          bmd[g] <= (v1k && !sh[10]) ? 17'(17'(e1.m[g]) << sh[4:0]) : 17'd0;
          bmdb[g] <= (pk1k && v1k && !shb[10]) ? 17'(17'(e1b.m[g]) << shb[4:0]) : 17'd0;
        end
      end
    end else begin : g_bs
      otpu_skew #(.W(8 * 17), .N(CLS - 1 + j)) u_bm (.clk, .en(1'b1), .d(bm2[j]), .q(bmd));
      if (WG > 1) begin : g_s2
        otpu_skew #(.W(8 * 17 + 2), .N(CLS - 1 + j)) u_bm2 (.clk, .en(1'b1),
                                                           .d({pk2 || w82, w82, bm2b[j]}),
                                                           .q({pkd, w8d, bmdb}));
      end else begin : g_n2
        assign {pkd, w8d, bmdb} = '0;
      end
    end
    otpu_skew #(.W(1), .N(CLS + 4 + j)) u_f (.clk, .en(1'b1), .d(m0.v && m0.first), .q(fst));
    // (WG > 1: one chain a group, the pre-adder's D int8's high halves)
    otpu_mxce #(.CPG((WG > 1) ? 2 : CPG)) u_ce (.clk, .a(ca), .d(cd),
                                                 .den((WG > 1) ? w8d : 1'b1), .bm(bmd),
                                                 .first(fst), .acc);
    otpu_skew #(.W(48), .N(MCOLS - 1 - j)) u_dsk (.clk, .en(1'b1), .d(acc), .q(acc_al[j]));
    if (WG > 1) begin : g_pk
      logic [7:0][23:0] ca2;
      logic [47:0]      acc2;
      always_comb
        for (int g = 0; g < 8; g++) begin
          logic [PRW-1:0] pe;
          pe = preg[j][g * CLS + CLS - 1];
          ca2[g] = {{8{pe[31]}}, pe[31:16]};
        end
      otpu_mxce #(.CPG(1)) u_ce2 (.clk, .a(ca2), .d('0), .den(1'b0), .bm(bmdb), .first(fst),
                                  .acc(acc2));
      otpu_skew #(.W(48), .N(MCOLS - 1 - j)) u_dsk2 (.clk, .en(1'b1), .d(acc2), .q(acc2_al[j]));
    end else begin : g_npk
      assign acc2_al[j] = '0;
    end
  end
  // PU: the pumped columns (otpu_mxpc) and each one's column ends for its two rows. Column 0
  // takes the ACT RAM's blocks a0 / a1 (S0), interleaves them per chain stage, and column j > 0
  // takes column j - 1's interleaved stream two fast cycles later; column j's weights are its
  // lane's (WA: the lane's bytes at S0 + j + ceil(k / 2), g_xp g_u's g_wa; else w0c[j] at S0);
  // row r's chain ends reach the column ends in cycle S0 + j + 10 + PHOLD + r, its B and first
  // with them (BRAW, both rows' offsets skewed with the entry's bytes; row 1's B a register
  // later), and its integers come out of the deskew at S0 + LR + r (acc_al / acc2_al row 0's)
  logic [MCOLS-1:0][47:0] acc_al1, acc2_al1;
  logic [MCOLS-1:0][D-1:0][7:0] pxs;        // the columns' streams (the hop: both rows interleaved)
  for (genvar j = 0; j < (PU ? MCOLS : 0); j++) begin : g_pcol
    logic [NCG-1:0][1:0]  wf_j;
    logic [NCG-1:0][31:0] q0, q1;
    for (genvar g = 0; g < NCG; g++) begin : g_wf
      assign wf_j[g] = wfc[j][g];
    end
    (* keep_hierarchy = "yes" *)         // (otpu_clk2x.tcl finds the columns' clocks by its pins)
    otpu_mxpc #(.D(D), .CLS(CLS), .COL(j), .HOLD(PHOLD), .WSK(WA)) u_pc (
      .clk, .clk_f, .rst,
      .x0p(a0[0 +: D*8]), .x1p(a1), .xp(pxs[(j == 0) ? 0 : j - 1]), .xh(pxs[j]),
      .w(WA ? wsk[j] : w0c[j]), .wfc(wf_j), .q0, .q1);
    praw_t       rw0, rwk;
    bdec_t       e1, e1b;
    logic [31:0] da1k, db1k;
    logic        v1k, pk1k, w81k, pkd0, w8d0, pkd1, w8d1, fst0, fst1;
    logic [7:0][16:0] bmd0, bmdb0, bmx1, bmxb1, bmd1, bmdb1;
    assign rw0 = '{b: '{r0: gent(gq0, j, 0, wfc[j][0]), r1: gent(gq0, j, 1, wfc[j][0]),
                        da: ascm[32 +: 32], wf: wfc[j][0], u: cu0 || rs0, v: m0.v, pg: pg0,
                        w8: wf0 == WF_W8, lo: sl0},
                   da1: ascm1[32 +: 32]};
    otpu_skew #(.W($bits(praw_t)), .N(j + 8 + 32'(PHOLD))) u_bm (.clk, .en(1'b1), .d(rw0),
                                                               .q(rwk));
    always_ff @(posedge clk) begin
      e1 <= bdec(rwk.b.r0, rwk.b.wf, rwk.b.u, rwk.b.lo);
      e1b <= bdec(rwk.b.r1, rwk.b.wf, rwk.b.u, rwk.b.lo);
      da1k <= rwk.b.da;
      db1k <= rwk.da1;
      v1k <= rwk.b.v;
      pk1k <= rwk.b.pg;
      w81k <= rwk.b.w8;
      pkd0 <= pk1k || w81k;
      w8d0 <= w81k;
      pkd1 <= pkd0;
      w8d1 <= w8d0;
      for (int g = 0; g < 8; g++) begin
        logic [10:0] sh, shb, sh1, shb1;
        sh = e1.s[g] - 11'(da1k[8 * (g / 2) +: 8]);
        shb = e1b.s[g] - 11'(da1k[8 * (g / 2) +: 8]);
        sh1 = e1.s[g] - 11'(db1k[8 * (g / 2) +: 8]);
        shb1 = e1b.s[g] - 11'(db1k[8 * (g / 2) +: 8]);
        bmd0[g] <= (v1k && !sh[10]) ? 17'(17'(e1.m[g]) << sh[4:0]) : 17'd0;
        bmdb0[g] <= (pk1k && v1k && !shb[10]) ? 17'(17'(e1b.m[g]) << shb[4:0]) : 17'd0;
        bmx1[g] <= (v1k && !sh1[10]) ? 17'(17'(e1.m[g]) << sh1[4:0]) : 17'd0;
        bmxb1[g] <= (pk1k && v1k && !shb1[10]) ? 17'(17'(e1b.m[g]) << shb1[4:0]) : 17'd0;
      end
      bmd1 <= bmx1;
      bmdb1 <= bmxb1;
      fst1 <= fst0;
    end
    otpu_skew #(.W(1), .N(j + 13 + 32'(PHOLD))) u_f (.clk, .en(1'b1), .d(m0.v && m0.first),
                                                   .q(fst0));
    for (genvar r = 0; r < 2; r++) begin : g_r
      logic [7:0][23:0] ca, cd, ca2;
      logic [47:0]      acc, acc2;
      wire              pkd = (r == 1) ? pkd1 : pkd0;
      always_comb
        for (int g = 0; g < 8; g++) begin
          logic [31:0] pe;
          pe = (r == 1) ? q1[g] : q0[g];
          ca[g] = pkd ? {{9{!pe[15]}}, pe[14:0]} : pe[23:0];
          cd[g] = {{4{pe[31]}}, pe[31:16], 4'd0};
          ca2[g] = {{8{pe[31]}}, pe[31:16]};
        end
      otpu_mxce #(.CPG(2)) u_ce (.clk, .a(ca), .d(cd), .den((r == 1) ? w8d1 : w8d0),
                                 .bm((r == 1) ? bmd1 : bmd0), .first((r == 1) ? fst1 : fst0),
                                 .acc);
      otpu_mxce #(.CPG(1)) u_ce2 (.clk, .a(ca2), .d('0), .den(1'b0),
                                  .bm((r == 1) ? bmdb1 : bmdb0), .first((r == 1) ? fst1 : fst0),
                                  .acc(acc2));
      if (r == 0) begin : g_a0
        otpu_skew #(.W(96), .N(MCOLS - 1 - j)) u_dsk (.clk, .en(1'b1), .d({acc2, acc}),
                                                     .q({acc2_al[j], acc_al[j]}));
      end else begin : g_a1
        otpu_skew #(.W(96), .N(MCOLS - 1 - j)) u_dsk (.clk, .en(1'b1), .d({acc2, acc}),
                                                     .q({acc2_al1[j], acc_al1[j]}));
      end
    end
  end
  // the row's last block, PAIR and M at S0 + LROW (PU: LR); Sw and the Sa' from S1. The valid
  // in reset registers (otpu_vdelay: a power-up one would push a row, below)
  logic             tl_v, tl_pr;
  logic [7:0]       tl_M;
  logic [31:0]      tl_sw;
  logic [MCOLS-1:0][31:0] tl_sa;
  otpu_vdelay #(.N(LR)) u_tlv (.clk, .rst, .en(1'b1), .d(m0.v && m0.last), .q(tl_v));
  otpu_delay #(.W(1 + 8), .N(LR)) u_tl (.clk, .en(1'b1), .d({pr0, M0}), .q({tl_pr, tl_M}));
  otpu_skew #(.W(32 + MCOLS * 32), .N(LR - 1)) u_ts (.clk, .en(1'b1), .d({sw1, sap1}),
                                                    .q({tl_sw, tl_sa}));
  // pg at S0 + LROW, and the Sw of rows 8..15 (WG > 1)
  logic             tl_pk;
  logic [MCOLS-1:0][31:0] tl_sab;
  if (WG > 1) begin : g_tlpk
    otpu_delay #(.W(1), .N(LR)) u_tlpk (.clk, .en(1'b1), .d(pg0), .q(tl_pk));
    otpu_skew #(.W(MCOLS * 32), .N(LR - 1)) u_tsb (.clk, .en(1'b1), .d(sap1b), .q(tl_sab));
  end else begin : g_ntlpk
    assign tl_pk = 1'b0;
    assign tl_sab = '0;
  end
  // PAIR: row j (j < M <= MCOLS/2) adds its partner column j + M's integer; pg: the row's 16
  // integers, row 2c + t column c's t-th sum, in two result rows (rows 0..7, then rows 8..15 a
  // cycle later: hb_*); registered for the result FIFO (rv: push)
  logic             rv, hb_v;
  logic [MCOLS-1:0][47:0] rowi, hb_i;
  logic [MCOLS-1:0][31:0] rsa, hb_sa;
  logic [31:0]      rsw;
  logic             rv_n;                   // rv's next value (the result FIFO copies' replicas)
  if (!PU) begin : g_rend
    assign rv_n = tl_v || (WG > 1 && hb_v);
    always_ff @(posedge clk) begin
      rv <= !rst && rv_n;
      hb_v <= !rst && WG > 1 && tl_v && tl_pk;
      for (int j = 0; j < MCOLS; j++) begin
        logic [47:0] pb;
        pb = '0;
        for (int m = 1; m <= MCOLS / 2; m++)
          if (j < MCOLS / 2 && j + m < MCOLS && tl_M == 8'(m)) pb = acc_al[j + m];
        if (WG > 1 && hb_v)
          rowi[j] <= hb_i[j];
        else if (WG > 1 && tl_pk)
          rowi[j] <= (j % 2 == 1) ? acc2_al[j / 2] : acc_al[j / 2];
        else
          rowi[j] <= (tl_pr && j < MCOLS / 2) ? acc_al[j] + pb : acc_al[j];
        hb_i[j] <= (j % 2 == 1) ? acc2_al[MCOLS / 2 + j / 2] : acc_al[MCOLS / 2 + j / 2];
      end
      rsa <= (WG > 1 && hb_v) ? hb_sa : tl_sa;
      hb_sa <= tl_sab;
      if (!(WG > 1 && hb_v)) rsw <= tl_sw;
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && WG > 1 && hb_v && tl_v)
        $fatal(1, "otpu_mxu: a row's end while a pg row's second half pushes");
`endif
  end else begin : g_prend
    // PU: a pass's rows end together, row 0's integers at S0 + LR (tl_v), row 1's a cycle later;
    // its result FIFO rows are pushed in the unpumped passes' order, a cycle each: row 0 (pg: its
    // rows 0..7, then 8..15), then row 1 (pg: likewise; none past an odd M, tl_x1). The first at
    // tl_v, the others from pq (codes: 1 row 0's 8..15, 2 int8's row 1 as it comes out of the
    // deskew, 3 / 4 pg's row 1's halves, registered as they come), so the drain sees the rows
    // of the unpumped MXU; the passes' last advances are at least their rows apart (pu_hold)
    logic             tl_x1, d1;
    logic [31:0]      tl_sw1, psw1;
    logic [2:0][2:0]  pq;
    logic [MCOLS-1:0][47:0] h1lo, h1hi;
    logic [MCOLS-1:0][31:0] psa;
    otpu_delay #(.W(1), .N(LR)) u_tlx (.clk, .en(1'b1), .d(x10), .q(tl_x1));
    assign rv_n = tl_v || pq[0] != 3'd0;
    otpu_skew #(.W(32), .N(LR - 1)) u_tsx (.clk, .en(1'b1), .d(sw1x), .q(tl_sw1));
    function automatic logic [MCOLS-1:0][47:0] half(input logic [MCOLS-1:0][47:0] a,
                                                    input logic [MCOLS-1:0][47:0] a2,
                                                    input int h);
      for (int j = 0; j < MCOLS; j++)
        half[j] = (j % 2 == 1) ? a2[h * MCOLS / 2 + j / 2] : a[h * MCOLS / 2 + j / 2];
    endfunction
    always_ff @(posedge clk) begin
      rv <= 1'b0;
      hb_v <= 1'b0;
      d1 <= !rst && tl_v;
      if (d1) begin                                   // (row 1's integers, a cycle after tl_v)
        h1lo <= half(acc_al1, acc2_al1, 0);
        h1hi <= half(acc_al1, acc2_al1, 1);
      end
      if (rst) begin
        pq <= '0;
      end else if (tl_v) begin
        rv <= 1'b1;
        rowi <= tl_pk ? half(acc_al, acc2_al, 0) : acc_al;
        rsa <= tl_sa;
        rsw <= tl_sw;
        hb_i <= half(acc_al, acc2_al, 1);
        hb_sa <= tl_sab;
        psa <= tl_sa;
        psw1 <= tl_sw1;
        pq <= tl_pk ? (tl_x1 ? {3'd4, 3'd3, 3'd1} : {3'd0, 3'd0, 3'd1})
                    : (tl_x1 ? {3'd0, 3'd0, 3'd2} : '0);
      end else if (pq[0] != 3'd0) begin
        rv <= 1'b1;
        case (pq[0])
          3'd1: begin rowi <= hb_i; rsa <= hb_sa; end
          3'd2: begin rowi <= acc_al1; rsa <= psa; rsw <= psw1; end
          3'd3: begin rowi <= h1lo; rsa <= psa; rsw <= psw1; end
          default: begin rowi <= h1hi; rsa <= hb_sa; rsw <= psw1; end
        endcase
        pq <= {3'd0, pq[2:1]};
      end
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && tl_v && pq[0] != 3'd0)
        $fatal(1, "otpu_mxu: a pass's end while the last pass's rows push (%0d)", pq[0]);
`endif
  end

  // ================================================================== result FIFO
  // Rows in LUT RAM; the drain reads the head row from flip-flops. rf_q trails the head by the
  // last cycle's pop (rf_hs: the head is rf_q + rf_hs), so the rows are read at rf_q, rf_q + 1
  // and rf_q + 2 (a row written this cycle bypassed) with no grant on the address; each cycle the
  // head row and the row after it are registered (rh0, rh1) and the last pop picks the head
  // (rfh). One RAM copy per read address: a single read port each infers as LUT RAM.
  // A row: MCOLS integers, MCOLS Sa', Sw
  localparam int RW_ = MCOLS * 48 + MCOLS * 32 + 32;
  logic [RFW-1:0] rf_q, rf_t;
  logic [RFW:0]   rf_n;
  logic           rf_hs;                     // the last cycle popped a row
  wire            rf_push = rv;
  logic [RW_-1:0] rowp, rh0, rh1, rfh;
  logic [2:0][RW_-1:0] rq;                   // rows rf_q + c, as they are after this cycle
  logic [2:0][RFW-1:0] rf_qc;
  always_comb begin
    rowp = {rsw, rsa, rowi};
    for (int c = 0; c < 3; c++) rf_qc[c] = rf_q + RFW'(c);
  end
  // each copy has its own write enable and address (replicas of rv and rf_t): one register
  // driving the three copies' write ports was the worst slow-clock path of the MX experiment's
  // 4-column OOC (docs/mx_mxu.md section 5, +0.329 ns). At RF_STYLE "block" the copies are block
  // RAMs read ahead (otpu_ram_1r1w_ra: at rf_q's next value, rf_qn, plus c; the RAM bypasses the
  // row written at the last edge, the copy the one written at this one). As LUT RAMs a copy was
  // some 1,100 cells whose every one each address bit reached (the card's m8: 37,632 address
  // pins), and the write took a kept wv / wt per column slice (r22's m8: one wt at fanout 1,125,
  // 6.7 ns of route)
  wire [RFW-1:0] rf_qn = rst ? '0 : RFW'(rf_q + RFW'(rf_hs));
`ifndef SYNTHESIS
  logic [RW_-1:0] msh [RF];                  // the rows as written (the checks)
  logic [RF-1:0]  msv;                       // ... since the reset (the RAMs' other rows: anything)
  always_ff @(posedge clk)
    if (rst) msv <= '0;
    else if (rf_push) begin msh[rf_t] <= rowp; msv[rf_t] <= 1'b1; end
`endif
  for (genvar c = 0; c < 3; c++) begin : g_rfm
    logic [RW_-1:0] rqm;                     // row rf_q + c as the RAM has it (to the last edge)
`ifndef SYNTHESIS
    logic [RW_-1:0] mh;                      // the row at the head (the check below)
`endif
    logic           wv;
    logic [RFW-1:0] wt;
    if (RF_STYLE == "distributed") begin : g_lut
      // RF_STYLE "distributed": the LUT RAM read at rf_q + c, its write's kept wv / wt. A 64-row
      // copy fills an eighth of its block RAMs, and the read ahead's bypass (W registers and a
      // W-bit select a copy) costs more LUTs than the LUT RAM took (the WG targets' slice, u_mxu read
      // ahead: +312 LUTs, +2,021 FF and 28.5 block RAM tiles)
      (* keep *) logic           kv;
      (* keep *) logic [RFW-1:0] kt;
      (* ram_style = "distributed" *) logic [RW_-1:0] m [RF];
      always_ff @(posedge clk) begin
        kv <= !rst && rv_n;
        if (rst) kt <= '0;
        else if (kv) kt <= kt + 1'b1;
      end
      always_ff @(posedge clk) if (!rst && kv) m[kt] <= rowp;
      assign wv = kv;
      assign wt = kt;
      assign rqm = m[rf_qc[c]];
    end else begin : g_bram
      always_ff @(posedge clk) begin
        wv <= !rst && rv_n;
        if (rst) wt <= '0;
        else if (wv) wt <= wt + 1'b1;
      end
      otpu_ram_1r1w_ra #(.W(RW_), .N(RF), .STYLE("block")) u_m (
        .clk, .we(!rst && wv), .wa(wt), .wd(rowp), .ra_n(RFW'(rf_qn + RFW'(c))), .rd(rqm));
    end
    assign rq[c] = (wv && wt == rf_qc[c]) ? rowp : rqm;
`ifndef SYNTHESIS
    assign mh = msh[RFW'(rf_q + RFW'(rf_hs))];
    always @(posedge clk) begin
      if (!rst && rst_seen && (wv != rf_push || wt != rf_t))
        $fatal(1, "otpu_mxu: result FIFO copy %0d's write is not rf_push / rf_t", c);
      if (!rst && rst_seen && msv[rf_qc[c]] && rqm != msh[rf_qc[c]])
        $fatal(1, "otpu_mxu: result FIFO copy %0d's read ahead is not row %0d", c, rf_qc[c]);
    end
`endif
  end
  always_ff @(posedge clk) begin
    rh0 <= rf_hs ? rq[1] : rq[0];           // the head (rf_q + rf_hs) as it is after this cycle
    rh1 <= rf_hs ? rq[2] : rq[1];           // the row after it
  end
  assign rfh = rf_hs ? rh1 : rh0;
  wire [MCOLS-1:0][47:0] rfh_i  = rfh[0 +: MCOLS * 48];
  wire [MCOLS-1:0][31:0] rfh_sa = rfh[MCOLS * 48 +: MCOLS * 32];
  wire [31:0]            rfh_sw = rfh[MCOLS * 80 +: 32];

  // ================================================================== drain
  // lanes this cycle: results dj .. dj+ncnt-1 of the head row, stopping at a bank conflict
  logic [31:0] drow [MCOLS];                 // replay: dad of the head row's group 0
  logic [31:0] dad [MCOLS];                  // head row's TMEM addresses: out + n + j * ors,
                                             // kept incrementally
  logic [MW-1:0] dj;
  logic [MW-1:0] ncnt;
  logic [7:0]    dg;                         // the group of the row being drained
  logic [7:0]    dg1;                        // dg + 1, kept alongside
  // The drain's step is registered, with its next value computed a cycle ahead (below, at the
  // head's entries): no compare or subtract between the head's entry and the pick. d_last: dg1
  // == c_G; ncnt = min(run, c_Mn - dj) and dln its lanes (k < ncnt); d_fin: the step drains the
  // group's last results (dj + ncnt == c_Mn). drain_go's counts are tested by registered flags
  // (rf_nz: rf_n != 0, rl_nz: rows_live != 0) for the same reason.
  logic          d_last, d_fin, rf_nz, rl_nz;
  logic [NL-1:0] dln;
  // GRP: the weight groups left (the row's, on), and whether the row's is the last (its results
  // Nl): its rows' results, every pass's; pg: dh, the pass's second result row (rows 8..15,
  // the last group's Nl1)
  logic [15:0]   dw_n;
  logic          dw_lst, dh;
  wire  [MW-1:0] c_Mn = GA ? (dw_lst ? (dh ? h.Nl1 : h.Nl) : MW'(MCOLS))
                          : (d_last ? h.Ml : MW'(MCOLS));                    // its results
  // The drain lanes' addresses, dad[dj + k], are registered (dal): set wherever dj or dad move
  logic [NL-1:0][31:0] dal;
  function automatic logic [NL-1:0][31:0] dlane_a(input logic [31:0] a [MCOLS],
                                                  input logic [MW-1:0] j);
    for (int k = 0; k < NL; k++) dlane_a[k] = (32'(j) + 32'(k) < MCOLS) ? a[MW'(j) + MW'(k)] : '0;
  endfunction
  logic [NL-1:0][47:0] dx_l;
  logic [NL-1:0][31:0] dsa_l;
  logic [NL-1:0][7:0]  dcol_l;
  // (the result FIFO's head row is the head command's while it has rows not yet drained)
  wire  drain_go = rf_nz && rl_nz && (!c_asc || al_st == 2'd2);
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && rst_seen && drain_go && dg1 != dg + 8'd1) $fatal(1, "otpu_mxu: dg1 %0d, dg %0d", dg1, dg);
`endif
  // Results j and j' of a row sit ors * (j' - j) words apart: the same bank iff that is a multiple
  // of LANES. So consecutive results are conflict-free in runs of LANES / gcd(ors, LANES)
  // (capped at NL), a per-command constant: the lane count is min(run, M - dj).
  // A step of a group with `left` results still to drain: {drains them all, lanes}
  function automatic logic [MW:0] dstep(input logic [MW-1:0] left, input logic [MW-1:0] run);
    return {left <= run, (run < left) ? run : left};
  endfunction
  function automatic logic [NL-1:0] dlanes(input logic [MW-1:0] cnt);
    for (int k = 0; k < NL; k++) dlanes[k] = 32'(k) < 32'(cnt);
  endfunction
`ifndef SYNTHESIS
  // the registered step is its definition from the head's entry, dg1, dj and the counts
  always @(posedge clk) if (!rst && rst_seen && q_n != 0) begin
    logic [MW-1:0] mn, lf;
    mn = GA ? (dw_lst ? (dh ? h.Nl1 : h.Nl) : MW'(MCOLS)) : (dg1 == c_G) ? h.Ml : MW'(MCOLS);
    lf = mn - dj;
    if (d_last != (dg1 == c_G) || ncnt != ((c_run < lf) ? c_run : lf) || dln != dlanes(ncnt) ||
        d_fin != (MW'(dj + ncnt) == mn))
      $fatal(1, "otpu_mxu: drain step d_last %0d ncnt %0d d_fin %0d (dg1 %0d G %0d dj %0d M %0d run %0d)",
             d_last, ncnt, d_fin, dg1, c_G, dj, mn, c_run);
  end
  always @(posedge clk) if (!rst && rst_seen && q_n != 0 && dal != dlane_a(dad, dj))
    $fatal(1, "otpu_mxu: the drain lanes' addresses are not dad[dj + k] (dj %0d)", dj);
  always @(posedge clk) if (!rst && rst_seen && (rf_nz != (rf_n != 0) || rl_nz != (rows_live != 0)))
    $fatal(1, "otpu_mxu: rf_nz %0d (rf_n %0d), rl_nz %0d (rows_live %0d)", rf_nz, rf_n, rl_nz, rows_live);
`endif
  // a lane's column (its RMAX running max, its ASCALE factor): the result's, or at WG > 1 the
  // pass's activation row m (dg: a result row's results are all against row m)
  always_comb begin
    dx_l = '0; dsa_l = '0; dcol_l = '0;
    for (int k = 0; k < NL; k++)
      if (32'(dj) + 32'(k) < MCOLS) begin
        dx_l[k] = rfh_i[MW'(dj) + MW'(k)];
        dsa_l[k] = rfh_sa[MW'(dj) + MW'(k)];
        dcol_l[k] = GA ? dg : 8'(dj) + 8'(k);
      end
  end
`ifndef SYNTHESIS
  // PAIR reads a block's data and its partner's in one line
  always @(posedge clk)
    if (!rst && rst_seen && a_req && i_pair && !i_sws && !i_rsc && scale_addr[2])
      $fatal(1, "otpu_mxu: PAIR scales must be 8-byte aligned");
  // the lane count is what a greedy pick that stops at the first bank conflict would take
  always @(posedge clk) if (!rst && rst_seen && drain_go) begin
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
  // (GRP: the next row's results by its weight group's, the next one's after a last pass)
  wire             dw_nl = d_last ? (dw_n == 16'd2) : dw_lst;
  wire [MW:0]      st_mid = dstep(d_rest, c_run);
  wire [MW:0]      st_grp = GA ? (dw_nl ? {h.dflg, h.dnlg} : dstep(MW'(MCOLS), c_run))
                          : d_nl ? {h.dfl, h.dnl} : dstep(MW'(MCOLS), c_run);
  // (pg: after a pass's first result row its second, the same pass and weight group)
  wire             d_h1 = (WG > 1) && h.pg && !dh;
  wire [MW:0]      st_h1 = dw_lst ? {h.dflg1, h.dnlg1} : dstep(MW'(MCOLS), c_run);
  wire [NL-1:0]    ln_mid = dlanes(st_mid[MW-1:0]), ln_grp = dlanes(st_grp[MW-1:0]),
                   ln_h1 = dlanes(st_h1[MW-1:0]);
  logic            ea;          // the epilogue's advance: the TMEM grant (TQ 0) or flip-flops'
  logic [1:0]      tq_n;        // TQ: the epilogue's TMEM writes waiting (the TMEM interface)
  wire             d_pop = ea && drain_row_done;      // a row leaves the result FIFO
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && rst_seen && rf_n != 0 && rfh != g_rfm[0].mh)
    $fatal(1, "otpu_mxu: registered head row is not row %0d", rf_q + RFW'(rf_hs));
`endif

  // ---- the epilogue lanes: the pick registered (P1), i2f48, x Sw, x Sa' (otpu_mx_lane), the
  // pick's lanes, addresses and columns alongside; everything advances with ea (the TMEM grant,
  // or with TQ a function of flip-flops: the TMEM interface below). The
  // lane output (lo_*) is the drain's TMEM request: a write, or ACC's read of the old values
  typedef struct packed {
    logic                   v;
    logic [NL-1:0]          m;
    logic [NL-1:0][31:0]    ad;
    logic [NL-1:0][7:0]     col;
  } lmeta_t;
  lmeta_t p1, lo;
  logic [NL-1:0][31:0] lo_ad;
  logic [NL-1:0][7:0]  lo_col;
  logic [NL-1:0][47:0] p1_x;
  logic [NL-1:0][31:0] p1_sa;
  logic [31:0]         p1_sw;
  f32_t                lo_y [NL];
  always_ff @(posedge clk)
    if (rst) p1 <= '0;
    else if (ea) begin
      p1.v <= drain_go;
      p1.m <= drain_go ? dln : '0;
      p1.ad <= dal;
      p1.col <= dcol_l;
    end
  always_ff @(posedge clk)
    if (ea) begin
      p1_x <= dx_l; p1_sa <= dsa_l; p1_sw <= rfh_sw;
    end
  // GRP (WG > 1): the row's column factors are the Sw and its row factor Sa': each lane's first
  // multiply takes its column's, the second the row's
  logic [NL-1:0][31:0] p1_swl, p1_sal;
  if (WG > 1) begin : g_lsw
    always_ff @(posedge clk)
      if (ea)
        for (int k = 0; k < NL; k++) begin
          p1_swl[k] <= dsa_l[k];
          p1_sal[k] <= rfh_sw;
        end
  end else begin : g_nlsw
    for (genvar k = 0; k < NL; k++) begin : g_k
      assign p1_swl[k] = p1_sw;
      assign p1_sal[k] = p1_sa[k];
    end
  end
  for (genvar k = 0; k < NL; k++) begin : g_lane
    otpu_mx_lane #(.LM(LM)) u_l (.clk, .en(ea), .x(p1_x[k]), .sw(p1_swl[k]), .sa(p1_sal[k]),
                                 .y(lo_y[k]));
  end
  // the picks' addresses and columns in a delay line; their valid and lane bits in reset
  // registers (a run stopped in the middle of a drain leaves no write to come out after rst)
  lmeta_t lo_p;
  otpu_delay #(.W($bits(lmeta_t)), .N(LLAT - 2)) u_lm (.clk, .en(ea), .d(p1), .q(lo_p));
  logic [LLAT-2:0][NL:0] lvm;
  always_ff @(posedge clk)
    if (rst) lvm <= '0;
    else if (ea) begin
      lvm[0] <= {p1.v, p1.m};
      for (int s = 1; s <= LLAT - 2; s++) lvm[s] <= lvm[s-1];
    end
  always_ff @(posedge clk)
    if (ea) begin
      lo_ad <= lo_p.ad;
      lo_col <= lo_p.col;
    end
  always_comb begin
    {lo.v, lo.m} = lvm[LLAT-2];
    lo.ad = lo_ad;
    lo.col = lo_col;
  end
  logic [3:0] ln_n;                          // picks in the lanes (any nonzero = busy)

  // read-modify-write pipeline for ACC: request now, read from the request register next cycle
  // (rA, TMEM's input stage), data the cycle after (r0), (old*alpha)+new, write
  typedef struct packed {
    logic                   v;
    logic [NL-1:0]          m;
    logic [NL-1:0][31:0]    ad, nv;
    logic [NL-1:0][7:0]     col;
  } rmw_t;
  rmw_t r0, rw;                               // r0: data arriving now; rw: at the write stage
  rmw_t rA;                                   // the read in the request register (t_ren)
  rmw_t rx;                                   // RMAX (no ACC): written lanes, compared next cycle
  f32_t ry [NL];
  // r1: r0 one granted cycle later, with the old values registered (xo): no path from the TMEM
  // block RAMs into the fmadd's DSP inputs in one cycle. The ASCALE factor (alr) is selected from
  // r0 as it moves into r1, so the fmadd's b operand is a flop too (no mux in front of the DSP);
  // c_asc and alpha cannot change while a valid entry sits in r0/r1 (the head changes only once
  // drained, alpha loads only before an ASCALE command drains)
  // r1's valid alone resets: its lanes, addresses and values are data, used only under it (rh;
  // u_rwv carries its lanes beside the valid, read only under rw.v), and xo / alr take no reset
  // in their enable
  rmw_t r1;
  f32_t xo [NL], alr [NL];
  always_ff @(posedge clk) begin
    if (ea) begin
      r1 <= r0;
      for (int k = 0; k < NL; k++) begin
        xo[k]  <= t_rdata[k];
        alr[k] <= c_asc ? alpha[r0.col[k][MW-2:0]] : F_ONE;
      end
    end
    if (rst) r1.v <= 1'b0;
  end
  for (genvar k = 0; k < NL; k++) begin : g_rmw
    otpu_fmadd #(.LM(LM), .LA(LA)) u_y (.clk, .en(ea), .a(xo[k]), .b(alr[k]), .c(r1.nv[k]),
                                        .y(ry[k]));
  end
  // the valid and lane bits in reset registers (otpu_vdelay), as lvm's: a run stopped in an ACC
  // leaves no write to come out after rst, and none comes from power-up
  localparam int RWB = $bits(rmw_t);
  logic [RWB-1:0] r1p, rwp;
  assign r1p = r1;
  otpu_vdelay #(.W(1 + NL), .N(LM + LA)) u_rwv (.clk, .rst, .en(ea), .d(r1p[RWB-1 -: 1 + NL]),
                                                .q(rwp[RWB-1 -: 1 + NL]));
  otpu_delay #(.W(RWB - 1 - NL), .N(LM + LA)) u_rw (.clk, .en(ea), .d(r1p[RWB-2-NL:0]),
                                                    .q(rwp[RWB-2-NL:0]));
  assign rw = rwp;
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
  // enable; reset, as its valids: a hit updates mk and mx_have); the last stage a flop, not an
  // SRL tap. c_rmax is fixed while entries are in flight (the head changes only once drained)
  logic [NL-1:0][MCOLS-1:0] rh, rh_p, rwh, rxh;
  always_comb
    for (int k = 0; k < NL; k++)
      for (int j = 0; j < MCOLS; j++) begin
        rh[k][j]  = r1.v && c_rmax && r1.m[k] && (r1.col[k][MW-2:0] == (MW-1)'(j));
        rxh[k][j] = rx.v && rx.m[k] && (rx.col[k][MW-2:0] == (MW-1)'(j));
      end
  otpu_vdelay #(.W(NL * MCOLS), .N(LM + LA - 1)) u_rwh (.clk, .rst, .en(ea), .d(rh), .q(rh_p));
  always_ff @(posedge clk) if (rst) rwh <= '0; else if (ea) rwh <= rh_p;
  // per column: the candidate key is selected by the hits alone. At most one lane hits a column
  // per cycle (lanes drain distinct columns) and rx / rw are exclusive (rx only for !c_acc
  // commands, rw only for c_acc ones). The compare is a stage later: each column's hit and key
  // are registered at the advance (mq_h, mq_k) and compared with mk, and applied, the cycle after,
  // advance or not (fkey of the fmadd's sum, the lanes' selection, a 32-bit compare and the
  // update's enable were one cycle: the m8 card at a 150 MHz equivalent, -0.057 ns, 10 levels).
  // The running max holds: each column's keys reach mk one at a time, in order, each compared
  // with the max of those before it; a command completes only with the stage empty (c_drained).
  // WG > 1: a drain step's lanes are one result row's, all against one activation row (their
  // column), so a stage first takes the lanes' keys, a pair's max each (ql), then their max
  // (a tree) is the column's candidate in mq. max is the keys' order, so the max over a row's
  // results in any order is the ISA's in n order (docs/isa.md MM RMAX).
  logic [MCOLS-1:0] mx_hit, mx_upd, mq_h, mq_hn;
  logic [31:0] mx_cand [MCOLS], mq_k [MCOLS], mq_kn [MCOLS];
  logic        ql_e;                        // (WG > 1) the pairs' stage holds keys
  always_comb
    for (int j = 0; j < MCOLS; j++) begin
      mx_hit[j] = 1'b0; mx_cand[j] = '0;
      for (int k = 0; k < NL; k++) begin
        if (rxh[k][j]) begin
          mx_hit[j] = 1'b1;
          mx_cand[j] = mx_cand[j] | fkey(rx.nv[k]);
        end
        if (rwh[k][j]) begin
          mx_hit[j] = 1'b1;
          mx_cand[j] = mx_cand[j] | fkey(ry[k]);
        end
      end
      mx_upd[j] = mq_h[j] && (!mx_have[j] || mq_k[j] > mk[j]);
    end
  if (WG > 1) begin : g_mxg
    localparam int NQ = (NL + 1) / 2;
    logic [NQ-1:0]        ql_v;             // a pair's lanes hit
    logic [NQ-1:0][31:0]  ql_k;             // ... and the larger key
    logic [MCOLS-1:0]     ql_oh;            // the row's column
    always_ff @(posedge clk)
      if (rst) begin
        ql_v <= '0; ql_oh <= '0;
      end else begin
        ql_v <= '0; ql_oh <= '0;
        if (ea) begin
          for (int q = 0; q < NQ; q++) begin
            logic [31:0] ka, kb;
            logic        va, vb;
            va = |rxh[2*q] || |rwh[2*q];
            ka = (|rxh[2*q]) ? fkey(rx.nv[2*q]) : fkey(ry[2*q]);
            vb = 1'b0; kb = '0;
            if (2 * q + 1 < NL) begin
              vb = |rxh[2*q + 1] || |rwh[2*q + 1];
              kb = (|rxh[2*q + 1]) ? fkey(rx.nv[2*q + 1]) : fkey(ry[2*q + 1]);
            end
            ql_v[q] <= va || vb;
            ql_k[q] <= (vb && (!va || kb > ka)) ? kb : ka;
          end
          for (int j = 0; j < MCOLS; j++)
            for (int k = 0; k < NL; k++) if (rxh[k][j] || rwh[k][j]) ql_oh[j] <= 1'b1;
        end
      end
    always_comb begin
      logic [31:0] t;
      logic        tv;
      tv = 1'b0; t = '0;
      for (int q = 0; q < NQ; q++)
        if (ql_v[q] && (!tv || ql_k[q] > t)) begin tv = 1'b1; t = ql_k[q]; end
      mq_hn = tv ? ql_oh : '0;
      for (int j = 0; j < MCOLS; j++) mq_kn[j] = t;
    end
    assign ql_e = |ql_v;
`ifndef SYNTHESIS
    // a step's lanes are one column's
    always_ff @(posedge clk)
      if (!rst && rst_seen && ea)
        for (int k = 0; k < NL; k++)
          if ((rxh[k] != '0 || rwh[k] != '0) && (rxh[k] | rwh[k]) != mx_hit)
            $fatal(1, "otpu_mxu: an RMAX step's lanes in columns %b and %b", rxh[k] | rwh[k],
                   mx_hit);
`endif
  end else begin : g_mxn
    always_comb begin
      mq_hn = ea ? mx_hit : '0;
      for (int j = 0; j < MCOLS; j++) mq_kn[j] = ea ? mx_cand[j] : mq_k[j];
    end
    assign ql_e = 1'b0;
`ifndef SYNTHESIS
    always_ff @(posedge clk)
      if (!rst && rst_seen && ea)
        for (int j = 0; j < MCOLS; j++) begin
          int n;
          n = 0;
          for (int k = 0; k < NL; k++) n = n + int'(rxh[k][j]) + int'(rwh[k][j]);
          if (n > 1) $fatal(1, "otpu_mxu: several RMAX updates of column %0d in one cycle", j);
        end
`endif
  end
  always_ff @(posedge clk) for (int j = 0; j < MCOLS; j++) mq_k[j] <= mq_kn[j];

  wire c_drained = c_act && !cl_ld && (pn || c_lz) && (rows_live == 0) && (rmw_n == 0) &&
                   (ln_n == 0) && !rA.v && !r0.v && !r1.v && !rx.v && mq_h == '0 && !ql_e;
  // RMAX writes start a cycle after the head has drained: mx_q registers the test, so the
  // drained compares (c_left == 0, rows_live, rmw_n, ...) are off the TMEM write request and the
  // grant it feeds. The test holds while drained (nothing new enters the head until it
  // completes, which needs mx_done), and it is false in the completing cycle (mx_done), so a new
  // head never sees a stale mx_q.
  logic mx_q;
  // mx_go (mx_q && !mx_done) is a register of its own (mx_gq, r2.1): it starts the MXU's TMEM
  // write request (the RMAX lanes), which the slice's write-bank arbiter turns into the grant;
  // from mx_done through the request, the arbiter, the grant's read enables and the TMEM's
  // replay logic was r2.1's m8 163ab46d's one failing path (-0.003 ns, 14 levels)
  logic mx_gq;
  wire mx_go     = mx_gq;
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && rst_seen && mx_gq != (mx_q && !mx_done))
    $fatal(1, "otpu_mxu: mx_gq %0d, mx_q %0d, mx_done %0d", mx_gq, mx_q, mx_done);
`endif
  wire c_fin     = c_drained && (!c_rmax || mx_done || c_tz) && tq_n == 2'd0;
  // the head completes (while the pop head may pop the next command's chunks)
  wire c_done    = c_fin;
  wire al_go     = c_act && c_asc && al_st == 2'd0;

  // RMAX lanes (mx_i + k < c_M), registered as the drain's; next: mx_i + LANES. Their TMEM
  // addresses h.mxo + mx_i + k are registered too (mxl, loaded from h.mxo while no RMAX write
  // runs: mx_go is never set in a head's first cycle), so no adder precedes the write request
  logic [LANES-1:0] mxm, mxm_nx;
  logic [LANES-1:0][31:0] mxl;
  always_comb for (int k = 0; k < LANES; k++) mxm_nx[k] = 32'(mx_i) + LANES + 32'(k) < 32'(c_M);
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst && rst_seen && q_n != 0)
    for (int k = 0; k < LANES; k++)
      if (mxm[k] != (32'(mx_i) + 32'(k) < 32'(c_M)))
        $fatal(1, "otpu_mxu: RMAX lane %0d: mxm %0d, mx_i %0d, M %0d", k, mxm[k], mx_i, c_M);
  always @(posedge clk) if (!rst && rst_seen && mx_go)
    for (int k = 0; k < LANES; k++)
      if (mxl[k] != h.mxo + 32'(mx_i) + 32'(k))
        $fatal(1, "otpu_mxu: RMAX lane %0d address %0h, mxo %0h, mx_i %0d", k, mxl[k], h.mxo, mx_i);
`endif

  // TMEM's input stage (release r2, docs/tmem_nocopy.md 7): the read request is a register
  // (rq_*), a stage of the epilogue (it advances with ea), so no unit logic (the ASCALE
  // addresses' adder and compare, the drain's lane mux) precedes TMEM's lane routing, ranks and
  // block RAM pins; the word arrives a cycle after TMEM reads it (rA, al_st 3)
  logic [LANES-1:0]       t_ren_c, rq_en;
  logic [LANES-1:0][31:0] t_raddr_c, rq_ad;
  logic [LANES-1:0]       tw_en;                 // this cycle's writes
  logic [LANES-1:0][31:0] tw_ad, tw_d;
  always_ff @(posedge clk)
    if (rst) rq_en <= '0;
    else if (ea) begin rq_en <= t_ren_c; rq_ad <= t_raddr_c; end
  always_comb begin
    t_ren_c = '0; t_raddr_c = '0; tw_en = '0; tw_ad = '0; tw_d = '0;
    if (lo.v) begin
      for (int k = 0; k < NL; k++) begin
        if (c_acc) begin
          t_ren_c[k] = lo.m[k];
          t_raddr_c[k] = lo.ad[k];
        end else begin
          tw_en[k] = lo.m[k];
          tw_ad[k] = lo.ad[k];
          tw_d[k] = lo_y[k];
        end
      end
    end
    if (rw.v) begin
      for (int k = 0; k < NL; k++) begin
        if (rw.m[k]) begin
          tw_en[k] = 1'b1;
          tw_ad[k] = rw.ad[k];
          tw_d[k] = ry[k];
        end
      end
    end
    // ASCALE factors and RMAX values move LANES words per cycle (consecutive words: distinct
    // banks)
    if (al_go) begin
      for (int k = 0; k < LANES; k++) begin
        if (32'(al_i) + 32'(k) < 32'(c_M)) begin
          t_ren_c[k] = 1'b1;
          t_raddr_c[k] = c_asa + 32'(al_i) + 32'(k);
        end
      end
    end
    if (mx_go) begin
      for (int k = 0; k < LANES; k++) begin
        if (mxm[k]) begin
          tw_en[k] = 1'b1;
          tw_ad[k] = mxl[k];
          tw_d[k] = unkey(mk[MW'(32'(mx_i) + 32'(k))]);
        end
      end
    end
  end

  // ---- the TMEM interface. The epilogue gives TMEM, at each advance, the read in the request
  // register and this cycle's writes; it takes a read's word at the advance after the read's.
  // TQ 0: the slice grants the MXU's read and write ports as one group, and the epilogue
  // advances on the grant itself (ea = t_gnt), which reached every register of the epilogue in
  // the cycle the slice decides it (the m8 card: ~11.3 k enables at the end of the arbitration;
  // at a 150 MHz equivalent the MXU's worst paths, 720 endpoints within 0.2 ns).
  // TQ 1 (the slice's MXU_TQ: TMEM_IMPL 3 with replays): the reads are not granted. TMEM reads
  // what the MXU asks, as the VPU's (a read that loses a RAM is replayed the next cycle, t_lost),
  // and the grant takes only the writes, which wait for it in a queue of three (s: flip-flops,
  // the tail written every cycle it is free, so no grant or push reaches their enables; with
  // the queue empty TMEM takes the writes as they are issued). ea is then a function of
  // flip-flops, none of which the grant or t_lost reach in the cycle they decide: the room for
  // one more beat's writes, counted without this cycle's pop (room_q), no replay pending
  // (lost_q: the lost read's word comes the cycle after its replay), and, with a read in the
  // request register, no write waiting (tq_z: a read never passes an older write; it is read
  // with its advance's own writes, as one granted beat was). The request register is TMEM's
  // read address as it was (no logic before TMEM's lane routing). The grant reaches the queue's
  // counters only. A command completes once its writes have left the queue (c_fin).
  assign t_raddr = rq_ad;
  // a read's lanes by TMEM bank (bank b: the lane whose address's low bits are b; lanes of one
  // read hit distinct banks or read one word, so the AND-OR merge is exact): otpu_tmem's crossbar
  localparam int TBW = (TLANES > 1) ? $clog2(TLANES) : 1;
  initial if (TLANES < LANES) $fatal(1, "otpu_mxu: TLANES %0d < LANES %0d", TLANES, LANES);
  function automatic logic [TLANES-1:0] bk_en(input logic [LANES-1:0][31:0] ad,
                                              input logic [LANES-1:0] en);
    bk_en = '0;
    for (int b = 0; b < TLANES; b++)
      for (int l = 0; l < LANES; l++)
        if (en[l] && (TLANES == 1 || ad[l][TBW-1:0] == TBW'(b))) bk_en[b] = 1'b1;
  endfunction
  function automatic logic [TLANES-1:0][31:0] bk_ad(input logic [LANES-1:0][31:0] ad,
                                                    input logic [LANES-1:0] en);
    bk_ad = '0;
    for (int b = 0; b < TLANES; b++)
      for (int l = 0; l < LANES; l++)
        if (en[l] && (TLANES == 1 || ad[l][TBW-1:0] == TBW'(b))) bk_ad[b] = bk_ad[b] | ad[l];
  endfunction
  if (TQ) begin : g_tq
    typedef struct packed {
      logic [LANES-1:0]       wen;
      logic [LANES-1:0][31:0] waddr;
      logic [LANES-1:0][31:0] wdata;
    } wbeat_t;
    wbeat_t     s [3];                           // the waiting writes, a ring
    wbeat_t     hd;                              // its head
    logic [1:0] tq_h, tl;                        // the head's slot, the tail's (0..2)
    (* max_fanout = 64 *) logic room_q, tq_z, lost_q;
    logic       rq_any;                          // a read in the request register (|rq_en)
    logic [LANES-1:0] t_ren_q;                   // the read TMEM sees: rq_en while room_q, tq_z
    (* max_fanout = 128 *) wire ea_t = room_q && !lost_q && (tq_z || !rq_any);
    wire        rd   = room_q && !lost_q && tq_z;  // the request register's read happens now
    wire        iw   = ea_t && (|tw_en);           // writes issued
    wire        take = tq_z && t_gnt;              // ... and taken as issued
    wire        pop  = !tq_z && t_gnt;
    wire        push = iw && !take;
    wire [1:0]  tq_nn = tq_n + 2'(push) - 2'(pop);
    assign ea = ea_t;
    assign tl = (32'(tq_h) + 32'(tq_n) >= 3) ? 2'(32'(tq_h) + 32'(tq_n) - 3) : tq_h + tq_n;
    assign hd = s[tq_h];
    // TMEM's read enable is a register (its next value from rq_en's, room_q's and tq_z's: no LUT
    // in front of TMEM's read selection, whose conflicts decide the other ports' replays; the
    // m8 card at 150 MHz: room_q -> rd -> TMEM -> the VPU's hold, -0.041 ns). It leaves out
    // lost_q: in a replay's cycle the read stays up and the slice does not read it (rpl_frz, as
    // lost_q a register of TMEM's lost); the epilogue does not advance then, and reads it after
    assign t_ren = t_ren_q;
    // ... and by bank (t_rben / t_rbad, otpu_tmem PBR) from flip-flops too, decoded before the
    // registers from their next values (rb_v / rb_a: the request register's banks and their
    // lanes' addresses, loaded with rq_ad; rb_q: t_ren_q's banks), so TMEM's ranks, conflict
    // compares and block RAM addresses start at flip-flops, not at the lanes' bank decode (r150c,
    // the m8 card at a real 150 MHz: rq_ad -> the decode -> TMEM's compares -> lost_q, 11 levels,
    // -0.007 ns; rq_ad and rpv -> the decode -> the pumped RAMs' read pins, +0.03 to +0.04)
    logic [TLANES-1:0]       rb_v, rb_q;
    logic [TLANES-1:0][31:0] rb_a;
    assign t_rben = rb_q;
    assign t_rbad = rb_a;
    always_ff @(posedge clk) if (!rst && ea_t) rb_a <= bk_ad(t_raddr_c, t_ren_c);
    assign t_wen = tq_z ? (ea_t ? tw_en : '0) : hd.wen;
    assign t_waddr = tq_z ? tw_ad : hd.waddr;
    assign t_wdata = tq_z ? tw_d : hd.wdata;
    for (genvar i = 0; i < 3; i++) begin : g_s
      always_ff @(posedge clk)
        if (tq_n != 2'd3 && tl == 2'(i)) s[i] <= '{wen: tw_en, waddr: tw_ad, wdata: tw_d};
    end
    always_ff @(posedge clk)
      if (rst) begin
        tq_n <= '0; tq_h <= '0; tq_z <= 1'b1; room_q <= 1'b0; lost_q <= 1'b0; rq_any <= 1'b0;
        t_ren_q <= '0; rb_v <= '0; rb_q <= '0;
      end else begin
        tq_n <= tq_nn;
        tq_z <= tq_nn == 2'd0;
        if (pop) tq_h <= (tq_h == 2'd2) ? 2'd0 : tq_h + 2'd1;
        room_q <= 3'(tq_n) + 3'(iw) <= 3'd2;
        lost_q <= t_lost;
        if (ea_t) rq_any <= |t_ren_c;
        t_ren_q <= (ea_t ? t_ren_c : rq_en) & {LANES{3'(tq_n) + 3'(iw) <= 3'd2 && tq_nn == 2'd0}};
        if (ea_t) rb_v <= bk_en(t_raddr_c, t_ren_c);
        rb_q <= (ea_t ? bk_en(t_raddr_c, t_ren_c) : rb_v) &
                {TLANES{3'(tq_n) + 3'(iw) <= 3'd2 && tq_nn == 2'd0}};
      end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && rst_seen) begin
        if (push && tq_n == 2'd3) $fatal(1, "otpu_mxu: TMEM write queue overflow");
        if (rq_any != (|rq_en)) $fatal(1, "otpu_mxu: rq_any is not the request register's");
        if (tq_z != (tq_n == 2'd0)) $fatal(1, "otpu_mxu: tq_z is not the queue's");
        if ((|t_ren) && !tq_z) $fatal(1, "otpu_mxu: a read passes a waiting write");
        if (t_ren_q != (rq_en & {LANES{room_q && tq_z}}))
          $fatal(1, "otpu_mxu: t_ren_q is not the request register's read");
        // the read by bank is the request register's, decoded
        if (rb_v != bk_en(rq_ad, rq_en)) $fatal(1, "otpu_mxu: rb_v is not the request register's banks");
        if (rb_q != bk_en(rq_ad, t_ren_q)) $fatal(1, "otpu_mxu: rb_q is not t_ren_q's banks");
        for (int b = 0; b < TLANES; b++)
          if (rb_v[b] && rb_a[b] != bk_ad(rq_ad, rq_en)[b])
            $fatal(1, "otpu_mxu: rb_a[%0d] %0h is not the request register's %0h", b, rb_a[b],
                   bk_ad(rq_ad, rq_en)[b]);
        // TMEM loses only a read the MXU asked this cycle (and the slice took) or its replay
        if (t_lost && !((|t_ren) && rd) && !lost_q) $fatal(1, "otpu_mxu: TMEM lost a read not asked");
      end
`endif
  end else begin : g_ng
    assign ea = t_gnt;
    assign t_ren = rq_en, t_wen = tw_en, t_waddr = tw_ad, t_wdata = tw_d;
    assign t_rben = bk_en(rq_ad, rq_en), t_rbad = bk_ad(rq_ad, rq_en);
    assign tq_n = '0;              // (a lost read: the slice holds the grant for its replay)
  end

  // ASCALE factors: alpha takes the words every cycle they are there (al_st 3), not only at the
  // advance that moves al_st on. TMEM holds the MXU's read data until its next read (r1's xo
  // takes it there too; TQ: a lost read's lanes are wrong for a cycle, and the replay's word is
  // there by the advance) and al_i moves only with the advance, so after it alpha is what it was;
  // alpha is read (alr) only with the advance, and no ASCALE row is in r0 while it loads (drain_go
  // waits for al_st 2). Its MCOLS x 32 enables are off the grant's fan-out (the m8 card at a 150
  // MHz equivalent: 90 of the grant net's 127 failing endpoints). One write per entry (a constant
  // index): a variable-index write of several lanes makes Vivado try, and crash while
  // dissolving, a RAM for these MCOLS registers
  always_ff @(posedge clk)
    if (!rst && al_st == 2'd3)
      for (int j = 0; j < MCOLS; j++)
        if (32'(j) >= 32'(al_i) && 32'(j) < 32'(al_i) + LANES)
          alpha[j] <= t_rdata[$clog2(LANES)'(32'(j) - 32'(al_i))];
`ifndef SYNTHESIS
  // alpha_g: as loaded at the grant only (the old enables); the factor a row reads is its value
  logic [31:0] alpha_g [MCOLS];
  always_ff @(posedge clk)
    if (!rst && ea && al_st == 2'd3)
      for (int j = 0; j < MCOLS; j++)
        if (32'(j) >= 32'(al_i) && 32'(j) < 32'(al_i) + LANES)
          alpha_g[j] <= t_rdata[$clog2(LANES)'(32'(j) - 32'(al_i))];
  always @(posedge clk)
    if (!rst && rst_seen && ea && r0.v && c_asc)
      for (int k = 0; k < NL; k++)
        if (r0.m[k] && alpha[r0.col[k][MW-2:0]] != alpha_g[r0.col[k][MW-2:0]])
          $fatal(1, "otpu_mxu: ASCALE factor %0d is %h, at the grant %h", r0.col[k][MW-2:0],
                 alpha[r0.col[k][MW-2:0]], alpha_g[r0.col[k][MW-2:0]]);
`endif

  // ---- statistics for the profiler (per completed command): cycles the stream was starved
  // (work, no chunk), backpressured (chunks, no consumption), the drain frozen by the TMEM grant,
  // the issuer denied a DRAM port. The conditions are registered (st_c) and summed a cycle
  // late, off the grant paths: a command's count is st + g, g the last cycle's pending bit (none
  // after a command's end: that cycle's conditions belong to no command).
  logic [31:0] st_starve, st_bp, st_frz, st_deny;
  logic [3:0]  st_c, st_g;                  // {deny, frz, bp, starve}
  logic        st_f;                        // the last cycle ended a command
  assign st_g = st_f ? 4'd0 : st_c;

  // the issuer's next line compares (on the next addresses, registered)
  function automatic logic same_line(input logic [DAW-1:0] a, input logic [DAW-1:0] b);
    return a[DAW-1:5] == b[DAW-1:5];
  endfunction
  // (GRP: the 128-byte beat port A answers with)
  function automatic logic same_beat(input logic [DAW-1:0] a, input logic [DAW-1:0] b);
    return a[DAW-1:7] == b[DAW-1:7];
  endfunction

  always_ff @(posedge clk) begin
    done <= 1'b0;
    pf_u <= 1'b0;
    if (rst) begin
      i_act <= 1'b0; i_ini <= 1'b0; i_slot <= '0;
      q_n <= '0; bq_c <= '0; pn <= 1'b0; rows_p <= '0;
      occ <= '0; occz <= '0; socc <= '0; tg_n <= '0; tg_t <= '0; tg_h <= '0;
      f_head <= '0; f_tail <= '0; f_count <= '0; f_rd <= '0; f_tl <= '0; f_rdl <= '0;
      s_head <= '0; s_tail <= '0; s_count <= '0; s_rd <= '0;
      ck <= '0; cg <= '0; dg <= '0; dg1 <= 8'd1; c_left <= '0; c_lz <= 1'b1; cl_ld <= 1'b0; rows_live <= '0;
      c_l1 <= 1'b0; c_odd <= 1'b0; ckb <= '0;
      lk1 <= 1'b0; dh <= 1'b0; lq1 <= '0; lq2 <= '0; lq3 <= '0;
      rf_q <= '0; rf_hs <= 1'b0; rf_t <= '0; rf_n <= '0; rf_nz <= 1'b0; rl_nz <= 1'b0;
      dj <= '0; mx_done <= 1'b0; mx_have <= '0; mq_h <= '0;
      d_last <= 1'b0; d_fin <= 1'b0; ncnt <= '0; dln <= '0; mxm <= '0;
      al_st <= 2'd0; al_i <= '0; mx_i <= '0;
      r0 <= '0; rA <= '0; rx <= '0; rmw_n <= '0; ln_n <= '0;
      st_starve <= '0; st_bp <= '0; st_frz <= '0; st_deny <= '0;
      st_c <= '0; st_f <= 1'b0; mx_q <= 1'b0; mx_gq <= 1'b0;
      iq_n <= '0;
`ifndef SYNTHESIS
      // the oracle: a reset (a CHAIN reload) empties the FIFO, so the entries not yet accepted
      // stream again
      if (pf_on) pf_i <= pf_acc;
`endif
    end else begin
      logic [2:0] qn;
      qent_t bqx [BQD];                         // bq's next value
      int tc;
      logic bpush, bgo;
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
      bqx = bq; tc = int'(bq_c); bpush = 1'b0; bgo = 1'b0;
      cl_ld <= 1'b0;
      if (cl_ld) begin c_left <= h.total; c_lz <= h.tz; c_l1 <= h.t1; end
      // ---- accept a command (the head if the queue is empty, else the next entry); the issuer
      // loads its addresses in the next cycle (i_ini)
      if (start) begin
        if (q_n == 0) begin
          hn = cmd_e;                             // becomes the head now
          cl_ld <= 1'b1;
        end else if (q_n == 1 || QD == 2) nn = cmd_e;
        else bpush = 1'b1;                        // behind n: into bq (at the end)
        qn = qn + 1;
        if (!cmd_tz && !pf_on && !iq_on) begin
          i_ini  <= 1'b1;
          i_left <= cmd_total;
          i_KB   <= cmd_KBa;
          i_nv   <= cmd_nv;
          i_pg   <= cmd_pg;
          i_rs   <= cmd_e.kt ? {cmd.w5[31:1], 1'b0} : cmd.w5;    // (KT: the pair stride)
          i_kt   <= cmd_e.kt;
          i_spl  <= cmd_spl;
          i_hld  <= {cmd.w7[30:20], 10'd0};
          i_srs  <= cmd_srs;
          i_unit <= cmd.flags[0];
          i_rsc  <= cmd.flags[7] && !cmd.flags[0];
          i_wf   <= cmd.flags[5:4];
          i_w4   <= (cmd.flags[5:4] != WF_W8);
          i_pair <= cmd_pair;
          i_k0   <= cmd.w4[31:24];
          row_addr <= DAW'(cmd_a(cmd, 1)); chunk_addr <= DAW'(cmd_a(cmd, 1));
          srow_addr <= DAW'(cmd_a(cmd, 2));
        end
      end
`ifndef SYNTHESIS
      // the stream oracle: the issuer, idle, takes its next entry once its gate is passed
      if (pf_on && start && !cmd_tz) begin
        if (pf_acc >= pf_n || pf_tot[pf_acc] != cmd_total || pf_a1[pf_acc] != 64'(cmd_a(cmd, 1)))
          $fatal(1, "otpu_mxu: accept %0d (total %0d, address %h) is not the oracle's entry",
                 pf_acc, cmd_total, cmd_a(cmd, 1));
        pf_acc <= pf_acc + 1;
      end
      if (pf_on && !i_act && !i_ini && pf_i < pf_n && (pf_g[pf_i] < 0 || pf_acc > pf_g[pf_i])) begin
        i_ini  <= 1'b1;
        i_left <= pf_tot[pf_i];
        i_KB   <= 16'(pf_kba[pf_i]);
        i_nv   <= pf_nv[pf_i] != 0;
        i_pg   <= pf_pg[pf_i] != 0;
        i_rs   <= pf_fl[pf_i][16] ? {pf_w5[pf_i][31:1], 1'b0} : pf_w5[pf_i];  // (KT: the pair stride)
        i_kt   <= pf_fl[pf_i][16];
        i_spl  <= SPE && pf_w7[pf_i][31];
        i_hld  <= {pf_w7[pf_i][30:20], 10'd0};
        i_srs  <= (SPE && pf_w7[pf_i][31]) ? {12'd0, pf_w7[pf_i][19:0]} : pf_w7[pf_i];
        i_unit <= pf_fl[pf_i][0];
        i_rsc  <= pf_fl[pf_i][7] && !pf_fl[pf_i][0];
        i_wf   <= pf_fl[pf_i][5:4];
        i_w4   <= (pf_fl[pf_i][5:4] != WF_W8);
        i_pair <= pf_pr[pf_i] != 0;
        i_k0   <= pf_w4[pf_i][31:24];
        row_addr <= DAW'(pf_a1[pf_i]); chunk_addr <= DAW'(pf_a1[pf_i]);
        srow_addr <= DAW'(pf_a2[pf_i]);
        pf_i <= pf_i + 1;
      end
      if (pf_fd != 0 && start && !cmd_tz)
        $fwrite(pf_fd, "%0d %0d %0d %0d %0d %h %h %h %h %h %h\n", cmd_total, cmd_KBa, cmd_nv, cmd_pg,
                cmd_pair, cmd.w4, cmd.w5, cmd.w7, {15'd0, cmd_e.kt, 8'd0, cmd.flags},   // (bit 16: KT)
                64'(cmd_a(cmd, 1)), 64'(cmd_a(cmd, 2)));
`endif
      // IQ: the idle issuer takes the oldest queued issue (pushed a cycle or more before); an accept
      // pushes its own (behind the one taken this cycle)
      if (iq_on && !pf_on) begin
        int n;
        n = int'(iq_n);
        if (!i_act && !i_ini && n != 0) begin
          i_ini  <= 1'b1;
          i_left <= iq[0].left;
          i_KB   <= iq[0].KB;
          i_nv   <= iq[0].nv;
          i_pg   <= iq[0].pg;
          i_rs   <= iq[0].rs;
          i_kt   <= iq[0].kt;
          i_spl  <= iq[0].spl;
          i_hld  <= iq[0].hld;
          i_srs  <= iq[0].srs;
          i_unit <= iq[0].unit;
          i_rsc  <= iq[0].rsc;
          i_wf   <= iq[0].wf;
          i_w4   <= iq[0].w4;
          i_pair <= iq[0].pair;
          i_k0   <= iq[0].k0;
          row_addr <= iq[0].a1; chunk_addr <= iq[0].a1;
          srow_addr <= iq[0].a2;
          for (int j = 0; j < QD - 1; j++) iq[j] <= iq[j + 1];
          n = n - 1;
        end
        if (start && !cmd_tz) begin
          iq[n] <= '{left: cmd_total, rs: cmd_e.kt ? {cmd.w5[31:1], 1'b0} : cmd.w5, srs: cmd_srs, KB: cmd_KBa, k0: cmd.w4[31:24],
                     wf: cmd.flags[5:4], nv: cmd_nv, pg: cmd_pg, unit: cmd.flags[0],
                     rsc: cmd.flags[7] && !cmd.flags[0], w4: cmd.flags[5:4] != WF_W8,
                     pair: cmd_pair, kt: cmd_e.kt, spl: cmd_spl, hld: {cmd.w7[30:20], 10'd0},
                     a1: DAW'(cmd_a(cmd, 1)), a2: DAW'(cmd_a(cmd, 2))};
          n = n + 1;
        end
        iq_n <= ($clog2(QD)+1)'(n);
      end
      if (i_ini) begin
        logic [31:0] so;
        logic [DAW-1:0] sa, sn;
        logic [7:0]  ss;
        case (i_wf)
          WF_W8: begin so = 32'd4 + 32'(i_k0); ss = 5'd1; end
          WF_NV: begin so = 32'd16 + {21'd0, i_k0, 3'd0}; ss = i_pair ? 5'd16 : 5'd8; end
          default: begin so = 32'd8 + {22'd0, i_k0, 2'd0}; ss = i_pair ? 5'd8 : 5'd4; end
        endcase
        // GRP (WG 8): the group's Sw words (8 int8, 16 4-bit) padded to 32 / 64 / 128 bytes, then
        // a block's entries, a step's: 8 bytes (int8), two lines (int4x / fp4x6), four (NVFP4)
        if (GA)
          case (i_wf)
            WF_W8: begin so = 32'd32 + {21'd0, i_k0, 3'd0}; ss = 8'd8; end
            WF_NV: begin so = 32'd128 + {17'd0, i_k0, 7'd0}; ss = 8'd128; end
            default: begin so = 32'd64 + {18'd0, i_k0, 6'd0}; ss = 8'd64; end
          endcase
        sa = srow_addr + so;
        sn = srow_addr + i_srs;
        i_ini <= 1'b0;
        i_act <= 1'b1;
        i_k <= '0;
        i_klast <= (i_KB == 16'd1);
        i_l1 <= (i_left == 32'd1);
        i_odd <= 1'b0;
        i_tk <= 1'b0;
        i_soff <= so;
        i_sstep <= ss;
        scale_addr <= sa;
        srow_next <= sn;
        i_swd <= 1'b0;
        i_swh <= GA ? same_beat(srow_addr, sa) : same_line(srow_addr, sa);
        i_swn <= same_line(sa, sn);
        i_sws <= !i_unit && !i_rsc &&
                 (GA ? !same_beat(srow_addr, sa) : !same_line(srow_addr, sa));
      end
      // ---- release (in order): the head if not yet released, else the second entry
      if (go) begin
        if (q_n != 0 && !h.go) begin
          hn.go = 1'b1; hp.go = 1'b1;
        end else if (QD == 2 || q_n < 2 || !n.go) begin
          nn.go = 1'b1; np.go = 1'b1;
        end else begin                            // the oldest unreleased in bq, else the
          int k;                                  // command accepted this cycle
          k = -1;
          for (int i = BQD - 1; i >= 0; i--) if (i < tc && !bqx[i].go) k = i;
          if (k >= 0) bqx[k].go = 1'b1;
          else bgo = 1'b1;
        end
      end
      // ---- issue: the row's Sw line alone, or one advance (a chunk and its line)
      if (go_iss) begin
        if (i_sws) begin
          i_swd <= 1'b1;
          i_sws <= 1'b0;
        end else begin
          i_left <= i_left - 1;
          i_l1 <= (i_left == 32'd2);
          if (i_left == 1) i_act <= 1'b0;
          // KT: past a pair's first group its second's first advance (T_p, requested), the
          // second's chunks next to T_p; past the second the next pair at row_addr + rs
          i_tk <= KTE && i_klast && i_kt && !i_odd;
          if (i_klast) begin
            logic [DAW-1:0] sa, sn;
            sa = srow_next + i_soff;
            sn = srow_next + i_srs;
            i_k <= '0;
            i_klast <= (i_KB == 16'd1);
            i_odd <= KTE && i_kt && !i_odd;
            if (KTE && i_kt && !i_odd) begin
              chunk_addr <= chunk_addr + 32'(P);
            end else begin
              row_addr <= row_addr + i_rs;
              chunk_addr <= row_addr + i_rs;
            end
            srow_addr <= srow_next;
            srow_next <= sn;
            scale_addr <= sa;
            // the next row's Sw came with this line, or is in its first advance's, or needs a
            // request of its own
            i_swd <= a_tag.swr;
            i_swh <= GA ? same_beat(srow_next, sa) : same_line(srow_next, sa);
            i_swn <= same_line(sa, sn);
            i_sws <= !i_unit && !i_rsc &&
                     (GA ? !same_beat(srow_next, sa)
                         : (!a_tag.swr && !same_line(srow_next, sa)));
          end else begin
            logic [DAW-1:0] sa;
            sa = scale_addr + 32'(i_sstep);
            i_k <= i_k + 1;
            i_klast <= (i_k + 16'd2 == i_KB);
            // 4-bit: two blocks per chunk (PAIR: both in one advance); GRP: P bytes a chunk;
            // SPL: hi's chunk again (a triple's first), then lo's, HLD on
            if (end_c && !(KTE && i_tk))
              chunk_addr <= chunk_addr + ((SPE && i_spl) ? ((i_k[0] || !i_KB[0]) ? 32'(i_hld) : 32'd0)
                                                         : GA ? 32'(P) : 32'(D));
            scale_addr <= sa;
            i_swn <= same_line(sa, srow_next);
          end
        end
      end
      // ---- FIFO pushes (XPOST: the entries completed)
      if (XPOST ? f_cpl : f_we) begin
        f_tail <= f_tail + 1;
        for (int g = 0; g < FG; g++) f_tl[g] <= f_tl[g] + 1;
      end
      if (b_req || w_req) i_slot <= i_slot + 1'b1;
      if (a_req) tg_t <= tg_t + 1'b1;
      if (a_rvalid) tg_h <= tg_h + 1'b1;
      tg_n <= tg_n + (a_req ? 1'b1 : 1'b0) - (rv1 ? 1'b1 : 1'b0);
      if (r_cpl) s_tail <= s_tail + 1'b1;
      f_count <= f_count + (f_cpl ? 1 : 0) - (fpop_f ? 1 : 0);
      occ <= occ + ((b_req || w_req) ? 1'b1 : 1'b0) - (fpop_f ? 1'b1 : 1'b0);
      occz <= occz + ((b_req || w_req) ? 1'b1 : 1'b0) - (fpz ? 1'b1 : 1'b0);
      s_count <= s_count + (r_cpl ? 1 : 0) - ((fpop && !c_unit) ? 1 : 0);
      if (KTE)
        socc <= socc + ((go_iss && !i_sws && !i_unit) ? 1'b1 : 1'b0) -
                ((fpop && !c_unit) ? 1'b1 : 1'b0);
      // ---- pop one advance (the row's last group frees its chunk and scale entry; KT: an even
      // group's last keeps its chunk, T_p, for the odd group's first: ktt)
      if (fpop_f) f_head <= f_head + 1;
      if (fpop && !c_unit) s_head <= s_head + 1;
      if (pop) begin
        if (last_g) begin
          c_left <= c_left - 1; c_lz <= (c_left == 32'd1); c_l1 <= (c_left == 32'd2);
        end
        // the next read: the next entry, or back to the head for the row's next group
        if (last_k && !last_g) begin
          f_rd <= f_head;
          f_rdl <= {FG{f_head}};
          if (!c_unit) s_rd <= s_head;
        end else if (cdone) begin
          if (!ktt) begin
            f_rd <= f_rd + 1;
            for (int g = 0; g < FG; g++) f_rdl[g] <= f_rdl[g] + 1;
          end
          if (!c_unit) s_rd <= s_rd + 1;
        end
        if (ck == 0) begin                      // (pg: the row's two result rows; PU: c_np)
          if (!pn) rl = rl + (PU ? (RFW+1)'(c_np) : (c_pg ? 2 : 1));
          else rp = rp + (PU ? (RFW+1)'(c_np) : (c_pg ? 2 : 1));
        end
        ck <= last_k ? '0 : ck + 1;
        if (last_k) cg <= last_g ? '0 : cg + 1;
        // KT: the group after a pair's first is its second, unless the command ends there
        // SPL: block ab + 1 on a triple's second advance (lo.hi), else ab
        if (CKB) begin
          logic odd;
          odd = (last_k && last_g) ? c_kt && !c_odd && !c_l1 : c_odd;
          c_odd <= odd;
          ckb <= last_k ? (odd ? pop_e.KBm1 : '0) : c_odd ? ck :
                 (SPE && pop_e.spl) ? {15'd0, !ck[0] && pop_e.KB[0]} : ck + 16'd1;
        end
      end
      lk1 <= pop && last_k && c_pg;
      lq1 <= (PU && pop && last_k) ? c_np : 3'd0;
      lq2 <= lq1;
      lq3 <= lq2;
      // ---- a finished row enters the result FIFO
      if (rf_push) begin
        rf_t <= rf_t + 1;
        rn = rn + 1;
      end
      // ---- RMAX: the stage's keys into the running max (every cycle), the advance's hits into it
      for (int j = 0; j < MCOLS; j++) begin
        if (mx_upd[j]) mk[j] <= mq_k[j];
        if (mq_h[j]) mx_have[j] <= 1'b1;
      end
      mq_h <= mq_hn;
      // ---- drain (holds while the epilogue does not advance)
      if (ea) begin
        r0 <= rA;
        rA <= '0;
        rx <= '0;
        if (lo.v) begin
          if (c_acc) begin
            rA.v <= 1'b1;
            rA.m <= lo.m;
            rA.ad <= lo.ad;
            for (int k = 0; k < NL; k++) rA.nv[k] <= lo_y[k];
            rA.col <= lo.col;
          end
          if (c_rmax && !c_acc) begin
            rx.v <= 1'b1;
            rx.m <= lo.m;
            for (int k = 0; k < NL; k++) rx.nv[k] <= lo_y[k];
            rx.col <= lo.col;
          end
        end
        if (drain_go) begin
          if (drain_row_done && d_h1) begin         // pg: the pass's second result row next
            dj <= '0;
            dh <= 1'b1;
            for (int j = 0; j < MCOLS; j++) begin
              logic [31:0] a;
              a = dad[j] + 32'(MCOLS);
              dad[j] <= a;
              if (j < NL) dal[j] <= a;
            end
            rn = rn - 1;
            rl = rl - 1;
          end else if (drain_row_done) begin
            dj <= '0;
            dh <= 1'b0;
            if (d_last) begin                       // the next weight row, group 0
              dg <= '0; dg1 <= 8'd1;
              dw_n <= dw_n - 16'd1;                 // (GRP: the next weight group, R words on)
              dw_lst <= (dw_n == 16'd2);
              for (int j = 0; j < MCOLS; j++) begin
                logic [31:0] a;
                a = drow[j] + (GA ? (h.pg ? 32'(2 * MCOLS) : 32'(MCOLS)) : 32'd1);
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
        rmw_n <= rmw_n + ((lo.v && c_acc) ? 4'd1 : 4'd0) - (rw.v ? 4'd1 : 4'd0);
        ln_n <= ln_n + (drain_go ? 4'd1 : 4'd0) - (lo.v ? 4'd1 : 4'd0);
        if (mx_go) begin
          if (32'(mx_i) + LANES >= 32'(c_M)) mx_done <= 1'b1;
          else mx_i <= mx_i + 8'(LANES);
        end
        if (al_go) al_st <= 2'd1;
        if (al_st == 2'd1) al_st <= 2'd3;     // the read leaves the request register
        if (al_st == 2'd3) begin               // (the words into alpha: below)
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
        else if (ea && 32'(mx_i) + LANES < 32'(c_M)) mxl[k] <= mxl[k] + 32'(LANES);
      rows_live <= rl;
      rows_p <= rp;
      // rn != 0, rl != 0 from the counts' own zero / one tests: the grant enters last
      rf_nz <= rf_push || (rf_nz && !(rf_n == 1 && d_pop));
      rl_nz <= (pop && ck == 0 && !pn) || (rl_nz && !(rows_live == 1 && d_pop));
      // the pop head moves to the next command once the head's chunks have all popped, if the
      // pipeline treats both alike (PAIR, and M under PAIR: pr0, hi0 and M0 are read late; pg:
      // the packed PEs' select and its two result rows a row; WG > 1: the format, decoded at the
      // weights' skew's end)
      if (!pn && !cl_ld && c_lz && q_n >= 3'd2 && n.go && !c_done &&
          n.pair == h.pair && n.pg == h.pg && (!h.pair || n.M == h.M) &&
          (WG == 1 || n.wf == h.wf)) begin
        pnx = 1'b1;
        c_left <= n.total;
        c_lz <= n.tz;
        c_l1 <= n.t1;
      end
      // ---- statistics
      st_c <= {want_iss && !go_iss, !ea && (drain_go || lo.v || rw.v || mx_go || al_go),
               more && f_count != 0 && !pop, more && f_count == 0};
      st_f <= c_done;
      mx_q <= c_drained && c_rmax && !mx_done && !c_tz;
      // mx_q's next value and mx_done's (set by the last RMAX lanes' grant; a completing head
      // clears it, and then mx_q's next is low: it completes only with mx_done or c_tz)
      mx_gq <= c_drained && c_rmax && !mx_done && !c_tz &&
               !(ea && mx_go && 32'(mx_i) + LANES >= 32'(c_M));
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
          ck <= '0; cg <= '0; ckb <= '0; c_odd <= 1'b0;
          // the next head's chunk count: the queued entry's, or (cl_ld) the total of a command
          // accepted this cycle, a cycle later
          c_left <= (q_n >= 3'd2) ? n.total : '0;
          c_lz <= (q_n >= 3'd2) ? n.tz : 1'b1;
          c_l1 <= (q_n >= 3'd2) && n.t1;
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
          a = (start && q_n == 2'd1) ? cmd.w3 + cmd_e.jo[j] : n.out + n.jo[j];
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
          a = cmd.w3 + cmd_e.jo[j];
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
          if (QD > 2 && tc > 0) begin             // bq's oldest becomes the next entry
            nx = bqx[0];
            for (int i = 0; i < BQD - 1; i++) bqx[i] = bqx[i + 1];
            tc = tc - 1;
          end else if (bpush) begin
            nx = cmd_e;
            nx.go = bgo;
            bpush = 1'b0;
          end
        end else begin
          hx = hn; nx = nn; hpx = hp; npx = np;
        end
        h <= hx; n <= nx;
        // the drain's next step: a new head starts at its first row's group 0 (all of it if the
        // row is one group); a drain step moves on in the group, or to the next group / row
        if (c_done || (start && q_n == 0)) begin
          logic [MW:0] st;
          st = GA ? ((hx.Gw == 16'd1) ? {hx.dflg, hx.dnlg} : dstep(MW'(MCOLS), hx.run))
             : hx.g1 ? {hx.dfl, hx.dnl} : dstep(MW'(MCOLS), hx.run);
          d_last <= hx.g1;
          dw_n <= hx.Gw;
          dw_lst <= (hx.Gw == 16'd1);
          dh <= 1'b0;
          {d_fin, ncnt} <= st;
          dln <= dlanes(st[MW-1:0]);
          for (int k = 0; k < LANES; k++) mxm[k] <= 32'(k) < 32'(hx.M);
        end else begin
          if (ea && drain_go) begin
            if (!d_fin) begin
              {d_fin, ncnt} <= st_mid;
              dln <= ln_mid;
            end else if (d_h1) begin
              {d_fin, ncnt} <= st_h1;
              dln <= ln_h1;
            end else begin
              d_last <= d_nl;
              {d_fin, ncnt} <= st_grp;
              dln <= ln_grp;
            end
          end
          if (ea && mx_go && 32'(mx_i) + LANES < 32'(c_M)) mxm <= mxm_nx;
        end
        // the pop head's entry, from the entries, not from the command being accepted: a
        // command that becomes the head as it is accepted gets its entry a cycle later (cl_ld;
        // the pipeline is empty then, and nothing pops before c_left loads)
        pop_q <= pnx ? npx : hpx;
        pwf <= pnx ? npx.wf : hpx.wf;
      end
      pn <= pnx;
      q_n <= qn;
      if (bpush) begin
        bqx[tc] = cmd_e;
        bqx[tc].go = bgo;
        tc = tc + 1;
      end
      bq <= bqx;
      bq_c <= (QD > 2) ? 3'(tc) : '0;            // (QD 2: no bq, the RTL of QD 2's)
    end
  end
`ifndef SYNTHESIS
  initial if (QD < 2 || QD > 7) $fatal(1, "otpu_mxu: QD %0d (2..7: q_n is 3 bits)", QD);
  // the head / next mapping relies on the rdy contract (start only with a free entry); the tags
  // in flight never exceed TD (the issuer waits), nor the scale entries the chunk slots
  always @(posedge clk) if (!rst && rst_seen) begin
    if (start && q_n >= 3'(QD)) $fatal(1, "otpu_mxu: start with %0d commands queued", q_n);
    if (32'(bq_c) != ((q_n > 2) ? 32'(q_n) - 2 : 0))
      $fatal(1, "otpu_mxu: %0d in bq, %0d commands queued", bq_c, q_n);
    if (start && !pf_on && !iq_on && (i_act || i_ini)) $fatal(1, "otpu_mxu: start while the issuer streams");
    if (iq_on && 32'(iq_n) > 32'(q_n)) $fatal(1, "otpu_mxu: %0d issues queued, %0d commands", iq_n, q_n);
    if (q_n != 0 && !cl_ld && pop_q != (pn ? n : h))
      $fatal(1, "otpu_mxu: pop_q is not the pop head's entry");
    if (q_n != 0 && pwf != pop_q.wf) $fatal(1, "otpu_mxu: pwf is not pop_q's format");
    if (32'(tg_n) > TD) $fatal(1, "otpu_mxu: %0d port A requests in flight", tg_n);
    if (32'(s_count) > DEPTH) $fatal(1, "otpu_mxu: scale FIFO overflow (%0d)", s_count);
    if (KTE && (32'(socc) > DEPTH || s_count > socc))
      $fatal(1, "otpu_mxu: KT: %0d scale entries issued, %0d in the FIFO (DEPTH %0d)", socc,
             s_count, DEPTH);
    if (rv1 && tg_n == 0) $fatal(1, "otpu_mxu: a port A response without a request");
  end
  // the stream oracle's watchdog: no accept and no pop for +mxwd cycles (default 2M) while the
  // replay's commands remain: its state, then $fatal (a replay that deadlocks ends, not runs on)
  int unsigned pf_idle, pf_wd;
  initial begin
    pf_idle = 0;
    if (!$value$plusargs("mxwd=%d", pf_wd)) pf_wd = 2000000;
  end
  always @(posedge clk) if (!rst && pf_on) begin
    if (start || pop || pf_acc == 0 || (pf_acc >= pf_n && q_n == 0)) pf_idle <= 0;
    else pf_idle <= pf_idle + 1;
    if (pf_idle == pf_wd) begin
      $display("otpu_mxu %0d: oracle stalled: entry %0d of %0d (gate %0d), accepts %0d, q_n %0d, h.go %0d",
               SID, pf_i, pf_n, (pf_i < pf_n) ? pf_g[pf_i] : -2, pf_acc, q_n, h.go);
      $display("  issuer: ini %0d act %0d left %0d occ %0d tg_n %0d; FIFO f %0d s %0d", i_ini, i_act,
               i_left, occ, tg_n, f_count, s_count);
      $display("  consumer: more %0d p_act %0d c_left %0d c_rch %0d row_in %0d f_av %0d s_av %0d ck %0d cg %0d rows %0d/%0d",
               more, p_act, c_left, c_rch, row_in, f_av, s_av, ck, cg, rows_live, rows_p);
      $fflush();
      $fatal(1, "otpu_mxu %0d: stream oracle stalled", SID);
    end
  end
`endif

endmodule

// The pumped packed column (otpu_mxu PUMP, WG > 1; the out-of-context test rtl/exp/pump_us's
// otpu_ppk_col without its column ends): column COL's D otpu_ppe at clk_f = 2 x clk make, in a
// slow cycle, the packed products of TWO activation rows (x0, x1: a pass's rows 2p and 2p + 1)
// against the column's weight chunk, so the column does two unpumped passes' work. The fabric
// stays at clk except, per position, the activation's 2:1 select and its register (b_f) and the
// phase toggles; clk_f rises with every edge of clk (the MMCM's, phase-aligned).
// Fast edges f (the slow edges the even f; t the chunk's cycle at the column's inputs, x0p / x1p
// and w; k = i % CLS the chain stage): P_k takes row r's running sum at f = 6 + k + r + 2 (t +
// COL), the cascade a stage a fast cycle and the rows a fast cycle apart; back through M (f - 1),
// B and AD (f - 2), b_f (f - 3). AD loads at the edges of k's parity and holds two fast cycles;
// A2 / D (the decoded weights) load on the slow edges from the weights skewed ceil(k / 2) + COL
// (WSK: w is that skew's output, otpu_mxu's WASK lanes), a two-fast-cycle path from clk; b_f
// takes x0 on the edges of k's other parity and x1 on k's parity, from skews of floor(k / 2) + 1
// (x0) and floor((k + 1) / 2) + 1 (x1) at column 0 (the others hop column COL - 1's a cycle
// later): its captures on odd edges are one-fast-period paths from clk. pe (1 on the even edges)
// and po: a toggle of each per pair of positions (kept, reset-aligned as rst falls), so no phase
// net fans out beyond two DSPs.
// Chain ends: row r's P_15 at f = 21 + r + 2 (t + COL). HOLD 0: q0 is P (taken on the slow edge
// after: one fast period), q1 a register loaded on the odd edge after its P; HOLD 1: both held,
// every return into clk a two-fast-cycle path, a slow cycle later. Row r's sums are at the
// column end's input registers in cycle t + COL + 10 + HOLD + r: otpu_mxu's packed sums, P = S0 +
// 2^15 + S1 * 2^16 (the chain's first stage adds the 2^15).
module otpu_mxpc #(
  parameter int D = 128,
  parameter int CLS = 16,
  parameter int COL = 0,            // the column (its skews: COL more; COL > 0 hops COL - 1's stream)
  parameter bit BF = 1'b1,          // b_f a fabric register (else the DSP's B1: BREG 2)
  parameter bit HOLD = 1'b1,        // the chain ends held into clk over two fast cycles
  parameter bit WSK = 1'b0          // w comes skewed: byte i in cycle t + ceil(k / 2) + COL
) (
  input  logic                    clk,
  input  logic                    clk_f,
  input  logic                    rst,        // the fast phase aligns as it falls
  input  logic [D-1:0][7:0]       x0p,        // COL 0: row 0's bytes, chunk t in cycle t
  input  logic [D-1:0][7:0]       x1p,        // COL 0: row 1's
  input  logic [D-1:0][7:0]       xp,         // COL > 0: column COL - 1's stream (its xh)
  output logic [D-1:0][7:0]       xh,         // the stream position k's B takes (clk_f)
  input  logic [D-1:0][7:0]       w,          // the column's lane: chunk t in cycle t (or WSK)
  input  logic [D/CLS-1:0][1:0]   wfc,        // the format, a copy per chain (otpu_mxu's wfc)
  output logic [D/CLS-1:0][31:0]  q0,         // row 0's chain ends (cycle t + COL + 10 + HOLD)
  output logic [D/CLS-1:0][31:0]  q1          // row 1's (a cycle later)
);
  localparam int NCH = D / CLS;
  localparam int NP = CLS / 2;
  initial if (CLS != 16 || D % CLS != 0) $fatal(1, "otpu_mxpc: chains of 16 (CLS %0d, D %0d)", CLS, D);
  localparam logic [1:0] WF_W8 = 2'd0;
  // otpu_mxu's decode (dec4, wdec): a byte's two 4-bit weights, or int8's nibbles
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
  function automatic logic [15:0] wdec(input logic [7:0] b, input logic [1:0] f);
    if (f == WF_W8) return {{4{b[7]}}, b[7:4], 4'd0, b[3:0]};
    return {dec4(b[7:4], f != 2'd1), dec4(b[3:0], f != 2'd1)};
  endfunction

  // the fast phase: per chain a reset copy, per pair of positions (and per chain end) a pe and a
  // po toggle. rst_c falls on a slow edge, rst_f a fast edge later (odd), where pe is loaded with
  // 1 (so it reads 1 on the even edges) and po with 0
  (* dont_touch = "true" *) logic rst_c;
  (* dont_touch = "true" *) logic [NCH-1:0] rst_f;
  (* dont_touch = "true" *) logic [NCH-1:0][NP-1:0] pe_t, po_t;
  (* dont_touch = "true" *) logic [NCH-1:0] pe_h, po_h;
  always_ff @(posedge clk) rst_c <= rst;
  always_ff @(posedge clk_f) begin
    rst_f <= {NCH{rst_c}};
    for (int c = 0; c < NCH; c++) begin
      pe_t[c] <= rst_f[c] ? {NP{1'b1}} : ~pe_t[c];
      po_t[c] <= rst_f[c] ? {NP{1'b0}} : ~po_t[c];
      pe_h[c] <= rst_f[c] ? 1'b1 : ~pe_h[c];
      po_h[c] <= rst_f[c] ? 1'b0 : ~po_h[c];
    end
  end

  logic [NCH-1:0][31:0] pl;                        // the chain ends' P
  for (genvar c = 0; c < NCH; c++) begin : g_ch
    logic [47:0] pc [CLS];
    for (genvar k = 0; k < CLS; k++) begin : g_k
      localparam int I = c * CLS + k;
      localparam bit ODD = (k % 2) == 1;
      // the weights: the lane's byte skewed ceil(k / 2) + COL, decoded into A2 / D
      logic [7:0] ws;
      if (WSK) begin : g_ws
        assign ws = w[I];
      end else begin : g_w
        otpu_skew #(.W(8), .N((k + 1) / 2 + COL)) u_w (.clk, .en(1'b1), .d(w[I]), .q(ws));
      end
      wire [15:0] wd = wdec(ws, wfc[c]);
      wire sel = ODD ? po_t[c][k / 2] : pe_t[c][k / 2];
      // the activation: column 0 skews the rows' bytes and takes x1 on the edges of k's parity,
      // x0 on the others (xs, B's input: through b_f under BF); column COL > 0 takes column
      // COL - 1's B input two fast edges (a slow cycle) later, the mux in column 0 alone
      logic [7:0] xs;
      if (COL == 0) begin : g_sk
        logic [7:0] x0s, x1s;
        otpu_skew #(.W(8), .N(k / 2 + 1))       u_x0 (.clk, .en(1'b1), .d(x0p[I]), .q(x0s));
        otpu_skew #(.W(8), .N((k + 1) / 2 + 1)) u_x1 (.clk, .en(1'b1), .d(x1p[I]), .q(x1s));
        wire [7:0] xm = sel ? x1s : x0s;
        if (BF) begin : g_bf
          logic [7:0] b_f;
          always_ff @(posedge clk_f) b_f <= xm;
          assign xs = b_f;
        end else begin : g_b1
          assign xs = xm;
        end
      end else begin : g_hop
        logic [7:0] h1, h2;
        always_ff @(posedge clk_f) begin
          h1 <= xp[I];
          h2 <= h1;
        end
        assign xs = h2;
      end
      assign xh[I] = xs;
      logic [31:0] pk;
      otpu_ppe #(.FIRST(k == 0), .BF(BF)) u_ppe (
        .clk_f, .ce_w(pe_t[c][k / 2]), .ce_ad(sel), .wlo(wd[7:0]), .whi(wd[15:8]), .act(xs),
        .pcin(k == 0 ? 48'd0 : pc[k == 0 ? 0 : k - 1]), .pcout(pc[k]), .p(pk));
      if (k == CLS - 1) begin : g_end
        assign pl[c] = pk;
      end
    end
  end

  // the chain ends into clk: row 0's and row 1's packed sums per chain
  if (HOLD) begin : g_hold
    logic [NCH-1:0][31:0] hold0, hold1, hold1e;
    always_ff @(posedge clk_f)
      for (int c = 0; c < NCH; c++) begin
        if (pe_h[c]) begin
          hold0[c] <= pl[c];
          hold1e[c] <= hold1[c];
        end
        if (po_h[c]) hold1[c] <= pl[c];
      end
    assign q0 = hold0;
    assign q1 = hold1e;
  end else begin : g_nhold
    logic [NCH-1:0][31:0] hold1;
    always_ff @(posedge clk_f)
      for (int c = 0; c < NCH; c++) if (po_h[c]) hold1[c] <= pl[c];
    assign q0 = pl;
    assign q1 = hold1;
  end
endmodule

// A shared epilogue lane (docs/mx_mxu.md 4): one result a cycle, once per output:
//   v  = i2f48(x)            round to nearest even (|x| < 2^47: normal or zero, no overflow)
//   y  = (v * sw) * sa       two otpu_fmul (sa: Sa * 2^-W, already scaled)
// Latency 4 + 2 LM: |x| | leading zeros | normalize | round | x sw (LM) | x sa (LM). Every
// register advances with en (the MXU's TMEM grant).
module otpu_mx_lane
  import otpu_fp::*;
#(parameter int LM = 2) (
  input  logic         clk,
  input  logic         en,
  input  logic [47:0]  x,
  input  f32_t         sw,
  input  f32_t         sa,
  output f32_t         y
);
  function automatic logic [5:0] lzc48(input logic [47:0] m);
    if (m[47:16] != 0) return lzc32(m[47:16]);
    return 6'd32 + 6'(lzc32({m[15:0], 16'd0}));
  endfunction
  logic        s1, z1, s2, z2, s3, z3;
  logic [47:0] g1, g2, g3;
  logic [5:0]  lz2, lz3;
  f32_t        fv;
  always_ff @(posedge clk) if (en) begin
    // L1: sign, magnitude
    s1 <= x[47];
    z1 <= (x == 0);
    g1 <= x[47] ? (~x + 48'd1) : x;
    // L2: leading zeros
    s2 <= s1; z2 <= z1; g2 <= g1;
    lz2 <= lzc48(g1);
    // L3: normalize (the leading one at bit 47)
    s3 <= s2; z3 <= z2; lz3 <= lz2;
    g3 <= g2 << lz2;
    // L4: round to 24 bits, nearest even
    begin
      logic [23:0] mm;
      logic [24:0] mr;
      logic g, st;
      logic [8:0] e;
      e = 9'd174 - 9'(lz3);                     // 127 + 47 - lz
      mm = g3[47:24];
      g = g3[23];
      st = |g3[22:0];
      mr = {1'b0, mm} + ((g && (st || mm[0])) ? 25'd1 : 25'd0);
      if (mr[24]) e = e + 9'd1;                 // mr == 2^24: its fraction bits are 0
      fv <= z3 ? F_ZERO : {s3, e[7:0], mr[22:0]};
    end
  end
  f32_t swd, sad, y1;
  otpu_delay #(.W(32), .N(4)) u_sw (.clk, .en, .d(sw), .q(swd));
  otpu_delay #(.W(32), .N(4 + LM)) u_sa (.clk, .en, .d(sa), .q(sad));
  otpu_fmul #(.LAT(LM)) u_m1 (.clk, .en, .a(fv), .b(swd), .y(y1));
  otpu_fmul #(.LAT(LM)) u_m2 (.clk, .en, .a(y1), .b(sad), .y);
endmodule

// A lane's posted beats staged across the clocks (otpu_mxu g_xp's g_u: the chunk FIFO's lanes in
// UltraRAM, which has one clock): four quarter RAMs of QW bits and N entries in LUT RAM (rtl/tech
// otpu_ram_1r1w_2c), quarter q written on wclk at wa[q] (we[q]: the channel's quarter q, or the
// whole beat), all four read on rclk at ra into the read registers (rd, re). Its one crossing,
// the RAMs' outputs into the read registers, is boards/bcu1525/constraints/otpu_mxu_stage.tcl's,
// scoped to this module (kept through synthesis: KEEP_HIERARCHY where the MXU builds it) by its
// cells' names: the RAMs g_q[q].u_q/g_lut.mem_reg*, the registers g_q[q].u_q/rd_reg[*].
module otpu_mxu_stage #(parameter int QW = 256, parameter int N = 32) (
  input  logic                         wclk,
  input  logic [3:0]                   we,
  input  logic [3:0][$clog2(N)-1:0]    wa,
  input  logic [3:0][QW-1:0]           wd,
  input  logic                         rclk,
  input  logic                         re,
  input  logic [3:0][$clog2(N)-1:0]    ra,
  output logic [4*QW-1:0]              rd
);
  for (genvar q = 0; q < 4; q++) begin : g_q
    otpu_ram_1r1w_2c #(.W(QW), .N(N), .STYLE("distributed")) u_q (
      .wclk, .we(we[q]), .wa(wa[q]), .wd(wd[q]), .rclk, .re, .ra(ra[q]), .rd(rd[QW*q +: QW]));
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
// (the systolic MXU's operand skews). SRL: every stage in the shift-register LUTs (Vivado keeps a
// flip-flop at either end of a chain from a block RAM otherwise), for a line whose user registers
// its output (the activation hop's)
module otpu_skew #(parameter int W = 8, parameter int N = 1, parameter bit SRL = 1'b0) (
  input  logic         clk,
  input  logic         en,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  if (N == 0) begin : g_wire
    assign q = d;
  end else if (SRL) begin : g_srl
    (* srl_style = "srl" *) logic [W-1:0] r [N];
    always_ff @(posedge clk) if (en) begin
      r[0] <= d;
      for (int k = 1; k < N; k++) r[k] <= r[k-1];
    end
    assign q = r[N-1];
  end else begin : g_regs
    logic [W-1:0] r [N];
    always_ff @(posedge clk) if (en) begin
      r[0] <= d;
      for (int k = 1; k < N; k++) r[k] <= r[k-1];
    end
    assign q = r[N-1];
  end
endmodule
