// One openTPU slice: sequencer (dispatch window + scoreboard), DMA, MXU + ACT RAM, quantizer,
// VPU and TMEM. Units run concurrently. Shared resources are arbitrated every cycle:
//   TMEM   each bank serves 3 reads + 1 write per cycle; units are granted all-or-nothing in
//          the priority order DMA, COLL, VPU, MXU drain, QUANT (a unit that is not granted holds)
//   DRAM B DMA first, then the MXU stream (read responses are routed back by tag); its outputs
//          come from a register (b_req its valid bit) and its read data enters through one
//   DRAM A the MXU scale stream (reads), its outputs from a register (a_req its valid bit); QST
//          writes have their own scalar write port (SW), whose outputs come from a register
//          (sw_req its valid bit)
// The slice's DRAM sits outside (otpu_top) so that board wrappers can swap it. The DRAM may
// refuse requests (a_rdy/b_rdy, which must not depend on this cycle's requests), return reads
// after any latency (in order per port), and acknowledge writes late (wr_idle: none pending);
// rd_idle: no read of the slice's in flight (a run stopped by the host, rst, leaves its reads
// to come back after it). A request in the SW, port A or port B register is in neither the units
// nor the DRAM, and port B's read data is the slice's a cycle after its b_rvalid: every write the
// slice gave is taken once wr_idle && !sw_req && !(b_req && b_we), and no read of the slice's is
// in flight once rd_idle && !a_req && !(b_req && !b_we) and there was no b_rvalid the cycle
// before (port A's requests are reads: a_we is 0).
//
// Loader: while the slice is held in reset (rst), ld_start copies ld_n instructions from DRAM
// byte address ld_addr (chunk aligned) into IMEM, one chunk (D/32 instructions) per cycle.
// It runs on sys_rst only, so the board can load a program and then release rst. Its reads
// start once the DRAM has no read in flight (rd_idle), so that no read of a stopped run, or of
// a load before, is taken for one of its rows; a LOAD restarts it. A HALT CHAIN reload that rst
// interrupts (the host stopped the run) stops.
//
// ICACHE (otpu_seq; docs/icache.md): IMEM is an instruction cache over the program in DRAM. A
// LOAD (and a HALT CHAIN's reload) copies nothing: once the DRAM has no read of the slice's in
// flight and no line fill is under way, it gives the sequencer the program's address and length
// and empties the cache (ic_inv). The sequencer's line fills are port B reads ahead of the DMA's
// and the MXU's, with the DMA's tag; the memory returns port B's reads in order, so a fill's
// chunks are told from the DMA's by their place in that order (ic_rv).
//
// HALT CHAIN (docs/isa.md): when the sequencer halts with CHAIN and the DRAM has no write
// pending (wr_idle), the slice holds the sequencer and the units in reset (c_rst), loads the
// chained program with the loader and releases them: the run goes on with the run's arguments
// in R8..R15 and TMEM and DRAM as they were. `halted` stays low meanwhile, and icount counts
// every program of the run. RLD, a collective-unit instruction, is served here: one TMEM word
// through the collective's read port, converted, to the sequencer (never sent to otpu_coll).
module otpu_slice
  import otpu_pkg::*;
#(
  parameter int SID        = 0,
  parameter int S          = 1,
  parameter int D          = 32,
  parameter int MCOLS      = 8,
  parameter int ACT_BLOCKS = 64,
  parameter int ACT_ROWS   = MCOLS,   // ACT RAM rows (> MCOLS: the MXU replays chunks)
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 16,
  parameter int FIFO_DEPTH = 128,
  parameter int LANES      = 8,
  parameter int WIN        = 32,
  parameter int RPB        = 4,       // TMEM reads per bank per cycle
  parameter int WPB        = 2,       // TMEM writes per bank per cycle
  parameter int MXU_IMPL   = 2,       // MXU dot product: 2 systolic, the only one (docs/mxu_systolic.md)
  parameter int MXU_CL     = 16,      // cascade chain length
  parameter int MXU_E      = 0,       // MXU epilogue lanes (0: MCOLS / 2; otpu_mxu E)
  parameter int MXU_QD     = 2,       // MXU commands queued (otpu_mxu QD; 2..7)
  parameter int WG         = 1,       // weight rows a group (otpu_mxu WG; docs/isa.md "Weight groups")
  parameter int QUANT_XD   = 1024,    // quantizer row buffer chunks (0: two passes over TMEM; otpu_quant XD)
  parameter bit QUANT_R1   = 1'b0,    // RSCALE's row factor read once a row (otpu_quant R1; timing only)
  parameter int QUANT_PW   = 1,       // a buffered QACT pass 1's chunks a cycle (otpu_quant PW; timing only)
  parameter bit QUANT_SPLIT = 1'b0,   // QACT / QST flag LO: the split's lo part (otpu_quant SPLIT; Config.SPLITQ)
  parameter bit QUANT_HL    = 1'b0,   // QACT flag HL: q and lo from one read (otpu_quant HL; Config.QHL)
  parameter int QUANT_SWW  = 1,       // the SW port's words: QST pass 1's 4 x QUANT_SWW bytes a cycle (otpu_quant SWW; timing only)
  parameter bit QUANT_QSTB  = 1'b0,   // QST rows through the row buffer (otpu_quant QSTB; timing only)
  parameter bit QUANT_QSTHL = 1'b0,   // QST HL: a GRP ROW QST's hi and lo from one read (otpu_quant QSTHL; Config.QSTHL)
  parameter bit QONCE      = 1'b0,    // QACT flag ST and LDA (otpu_quant, otpu_dma, otpu_actram DPORT, otpu_seq; Config.QONCE)
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,  // VPU lanes with the composite functions
  parameter int ULANES     = LANES,   // TMEM lanes of the MXU and the quantizer (<= LANES)
  parameter int PQ_WIN     = 64,      // cycles per P/Q counter window (+bucket= in simulation)
  parameter bit HAS_DSTEP  = 1'b1,    // streams (DSTEP, STREAM): SE's tail in the VPU
  parameter int TMEM_IMPL  = 0,       // TMEM: 0 a copy per read port (FPGA block RAM), 1 banked
                                      //   (three copies of 1R1W RAMs: an ASIC; otpu_tmem); 2
                                      //   (simulation only) the read arbitration of a candidate
                                      //   with one or two copies (otpu_tmem_model,
                                      //   docs/tmem_nocopy.md)
  parameter bit TMEM_RQ    = 1'b0,    // TMEM_IMPL 1: the banks' addresses from the requests by
                                      //   priority, and the arbiter grants by them (gnt_q)
  parameter int TMEM_NCP   = 3,       // TMEM_IMPL 1: copies (3; 2, 4, 5, 7: the read slots of
                                      //   pumped copies, docs/tmem_pump.md)
  parameter int TMEM_NSB   = 8,       // TMEM_IMPL 3 (EXPERIMENTAL): one copy, each bank TMEM_NSB
  parameter int TMEM_SBP   = 3,       //   RAMs, the sub-bank at word-address bit TMEM_SBP
                                      //   (docs/tmem_nocopy.md)
  parameter int TMEM_SBH   = 0,       // TMEM_IMPL 3: the sub-bank hashed (otpu_tmem SBH)
  parameter bit TMEM_RPL   = 1'b0,    // TMEM_IMPL 3: lost reads replayed (otpu_tmem RPL), the
                                      //   board's arbiter (ARB_MASK) with the replays' holds
  parameter bit TMEM_TDP   = 1'b0,    // TMEM_IMPL 3: true dual-port RAMs (otpu_tmem TDP), a
                                      //   unit's ports on different RAM ports (TM_SIDE)
  parameter bit TMEM_GST   = 1'b0,    // TMEM_IMPL 3: QUANT 3 and COLL the DMA's guests, as
                                      //   TMEM_IMPL 0's (otpu_tmem GST: one data select)
  parameter int TMEM_PMP   = 0,       // TMEM_TDP: 1 the RAMs pumped at 2x on clk_f (otpu_tmem
                                      //   PMP: a write takes no read port), 2 their model at the
                                      //   core clock (simulation)
  parameter string TMEM_STYLE = "block", // TMEM_PMP 1: the pumped RAMs' (otpu_tmem STYLE: "ultra")
  parameter string IMEM_STYLE = "auto",  // IMEM's RAMs (otpu_seq STYLE: "ultra")
  parameter string IMEM_WC_STYLE = "",   // its wait entries', if not IMEM_STYLE's (otpu_seq WSTYLE)
  parameter string MXU_RF_STYLE = "block", // the MXU's result FIFO (otpu_mxu RF_STYLE:
                                          //   "distributed")
  parameter string MXU_FIFO_STYLE = "block", // the MXU's chunk FIFO lanes (otpu_mxu FIFO_STYLE:
                                          //   "ultra")
  parameter bit MXU_WASK   = 1'b1,    // XPOST, UltraRAM lanes: read stage by stage (otpu_mxu WASK)
  parameter int MXU_RF_ROWS = 0,      // the MXU's result FIFO rows (otpu_mxu RF_ROWS; 0: its rule)
  parameter bit MXU_BRAW   = 1'b1,    // WG > 1: column ends' B from raw scale (otpu_mxu BRAW)
  parameter bit MXU_AASK   = 1'b1,    // WG > 1: ACT RAM read by chain stage (otpu_mxu AASK)
  parameter int ACT_UCOLS  = 0,       // the ACT RAM's columns in UltraRAM (otpu_actram UCOLS)
  parameter int TMEM_PRK   = 0,       // TMEM_IMPL 3: the read ports' ranks (3 bits a port, port p
                                      //   at bit 3p; 0: PRANK3, below)
  parameter int TMEM_SDM   = -1,      // TMEM_TDP: the read ports' RAM sides (bit p: port p on
                                      //   port A; -1: TM_SIDED, below)
  parameter bit SEQ_WC     = 1'b0,    // EXPERIMENTAL: otpu_seq's wait counts (WAITC, docs/seq_vpu.md)
  parameter int NCH        = 2,       // the DRAM's channels and their beat bytes (otpu_native_dram:
  parameter int CB         = 64,      //   port B's channel hash, port W's wide chunks of NCH x CB)
  parameter int QB         = 6,       // a QST's footprint in beats of 2^QB bytes (otpu_seq QB): the
                                      //   DRAM adapter's fill beat (otpu_board / otpu_top: CB where
                                      //   it fills partial SW beats and CB > 64; else 64 bytes)
  parameter int SE_R       = 0,       // streams on an engine of their own, SE_R rows a cycle
                                      //   (otpu_se_eng; 0: the VPU's engine; docs/se_engine.md)
  parameter int DMA_DEPTH  = 128,     // otpu_dma DEPTH: the LD chunk buffer, a DSTEP's read-ahead
  parameter int DMA_LDQ    = 1,       // otpu_dma LDQ: 2, an LD's DRAM reads under the command before it
  parameter int MXU_PUMP   = 0,       // the MXU's array pumped on clk_f (otpu_mxu PUMP)
  parameter int VPU_FSUM   = 0,       // otpu_vpu FSUM: EXP2SUB's fused row sums (Config.FSUM)
  parameter int VPU_SIG    = 0,       // otpu_vpu VSIG: VOP SIGMOID / SILU (Config.VSIG)
  parameter int VPU_MAC    = 0,       // otpu_vpu MAC: VOP MAC (Config.VMAC)
  parameter int VPU_OUTER1 = 0,       // otpu_vpu OUTER1: VOP OUTER decay one on MAC's (Config.OUTER1)
  parameter int VPU_CNS    = 3,       // otpu_vpu CNS: the composites' stages per lane (3, or 5: two passes)
  parameter bit OUTER      = 1'b1,    // VOP OUTER (otpu_vpu; 0: a VPU without it, Config.OUTER)
  parameter bit XPOST      = 1'b0,    // WG > 1: port W's beats posted into the MXU's chunk FIFO by
  parameter int XQB        = 0,       //   the channels (otpu_mxu XPOST, QB: 2^XQB writes a beat;
  parameter bit CHASH      = 1'b1,    //   CHASH, the channels' hash: QST GRP's lanes, otpu_quant)
  parameter bit DSTEP2     = 1'b0,    // DSTEP's pair (otpu_dma, otpu_se_tail; Config.DSTEP2)
  parameter int ST16       = 0,       // a stream's DRAM state in 16 bits: 1 e6m9, 2 bf16 (otpu_dma; Config.ST16)
  parameter bit ST16S      = 1'b0,    // with ST16: STREAM's flags ST16R / ST16W, one side alone (otpu_dma; Config.ST16S)
  parameter bit KT         = 1'b0,    // WG > 1: MM's K tail (otpu_mxu KT; Config.KT)
  parameter int MXU_SPL    = 0,       // WG > 1: MM SPL, a split's terms as one MM (otpu_mxu SPL; Config.SPLMM)
  parameter int AW         = 32,      // DRAM byte address bits (otpu_seq AW; 36 the BCU1525's)
  parameter bit ICACHE     = 1'b0,    // IMEM an instruction cache over the program (otpu_seq)
  parameter int IC_LINES   = 16,      // ICACHE: its lines (otpu_seq IC_LINES)
  parameter int IC_PF      = 2,       // ICACHE: the lines prefetched ahead (otpu_seq IC_PF)
  parameter int WA         = waw(AW)  // the ports' word address bits (derived: 32 at AW 32)
) (
  input  logic          clk,
  input  logic          clk_f,      // TMEM_PMP 1, MXU_PUMP: 2 x clk, rising with it (the MMCM's)
  input  logic          sys_rst,
  input  logic          rst,
  input  logic [31:0]   rinit [8],  // the run's arguments: R8..R15 at the start (otpu_seq)
  // program loader
  input  logic          ld_start,
  input  logic [31:0]   ld_addr,
  input  logic [31:0]   ld_n,
  output logic          ld_busy,
  // DRAM
  input  logic          a_rdy,
  input  logic          b_rdy,
  input  logic          sw_rdy,
  input  logic          wr_idle,
  input  logic          rd_idle,    // no read in flight (the loader waits for it)
  output logic          a_req,
  output logic          a_we,
  output logic [WA-1:0] a_addr,
  output logic [31:0]   a_wdata,
  output logic [3:0]    a_be,
  output logic          sw_req,     // scalar (QST) writes: one byte-enabled word per cycle
  output logic [WA-1:0] sw_addr,    //   (QUANT_SWW > 1: the aligned 4 QUANT_SWW bytes holding the
  output logic [32*QUANT_SWW-1:0] sw_wdata,  // word sw_addr, sw_be's)
  output logic [4*QUANT_SWW-1:0]  sw_be,
  input  logic          a_rvalid,
  input  logic [((WG > 1) ? CB : 32)*8-1:0] a_rdata,  // the aligned 32 bytes (WG > 1: the CB-byte
                                      //   beat) holding the word read (MM scale lines)
  output logic          b_req,
  output logic          b_tag,
  output logic          b_we,
  output logic [D/4-1:0] b_wmask,
  output logic [D*8-1:0] b_wdata,
  output logic [WA-1:0] b_addr,
  output logic [$clog2(NCH)-1:0] b_par,  // the fold of b_addr's wide chunk (chash: the channel
                                    //   hash, ^b_addr[WA-1:5] on the card; see the port mux)
  input  logic          b_rvalid,
  input  logic          b_rtag,
  input  logic [D*8-1:0] b_rdata,
  // port W (otpu_native_dram: P-byte reads, a wide chunk each, its beats lane by lane: lane c
  // channel c's, in its requests' order, the chunk's row c; the MXU's weight streams at WG > 1,
  // weight groups)
  input  logic          pw_rdy,
  output logic          pw_req,
  output logic [WA-1:0] pw_addr,    // word address (P aligned)
  input  logic [NCH-1:0] pw_bv,
  input  logic [NCH*CB*8-1:0] pw_bd,
  // XPOST (otpu_mxu, otpu_native_dram): each W read's chunk FIFO entry, one beat (a natural
  // chunk); the channels' posted beats into lane c of entry p_slot on p_clk, and the adapter's
  // counts of them (clk's)
  output logic [$clog2(FIFO_DEPTH)-1:0] pw_slot,
  output logic          pw_one,
  input  logic          p_clk,
  input  logic [NCH-1:0] p_we,
  input  logic [NCH-1:0][1:0][((XQB > 0) ? (1 << XQB) : 1)-1:0] p_q,     // (otpu_mxu's copies)
  input  logic [NCH-1:0][4:0][$clog2(FIFO_DEPTH)-1:0] p_slot,
  input  logic [NCH-1:0][(D*8 >> XQB)-1:0] p_d,
  input  logic [NCH-1:0][$clog2(FIFO_DEPTH)+1:0] p_cnt,
  // collective
  output logic                    coll_req,
  output cmd_t                    coll_cmd,
  input  logic                    coll_ack,
  input  logic [LANES-1:0]        coll_ren,
  input  logic [LANES-1:0][31:0]  coll_raddr,
  output logic [LANES-1:0][31:0]  coll_rdata,
  input  logic [LANES-1:0]        coll_wen,
  input  logic [LANES-1:0][31:0]  coll_waddr,
  input  logic [LANES-1:0][31:0]  coll_wdata,
  input  logic                    coll_wtop,   // FOLLOW: a local write advances f_top (1 at NC 1)
  output logic                    coll_gnt_local,
  input  logic                    coll_gnt,
  // status
  output logic          halted,
  output logic          error,
  output logic          wait_to,    // the error is a WAITW timeout (STATUS WAIT_TO)
  output logic          a_inval,    // a WAITW held: port A's held beats may be stale (otpu_dma ww_held)
  output logic [31:0]   icount,
  output perf_t         pf,         // activity and trace events, a cycle late (otpu_pkg)
  input  logic          dump
);
  localparam int BW = $clog2(LANES);
  localparam int CL = $clog2(NCH), P = NCH * CB;    // port W's wide chunk: P bytes (P > D: the port)
  // the memory's channel hash of a word address (otpu_native_dram fold): the XOR fold of its wide
  // chunk's index (bits WA - 1 .. log2(P / 4)) into log2(NCH) bits; at NCH 2 the parity
  function automatic logic [CL-1:0] chash(input logic [WA-1:0] word_addr);
    logic [CL-1:0] h;
    h = '0;
    for (int i = $clog2(P / 4); i < WA; i++) h[(i - $clog2(P / 4)) % CL] ^= word_addr[i];
    return h;
  endfunction
  // the DRAM's SW ready and write idle as the units and the arbiters see them: with the SW port's
  // output register (see there)
  logic sw_rdy_i, wr_idle_i;
  // port B as the units, the loader and the arbiters see it: through its output register (b_rdy_i,
  // rd_idle_i, wr_idle_i) and its input register (b_rvalid_i, b_rtag_i, b_rdata_i); see "port B's
  // registers"
  logic b_rdy_i, rd_idle_i, b_rvalid_i, b_rtag_i;
  logic [D*8-1:0] b_rdata_i;
  // port A's ready as the MXU sees it: through its output register (see "port A's register")
  logic a_rdy_i;
  // the stream engine (docs/stream.md): the DMA moves a stream (DSTEP, STREAM) through SE in
  // the VPU; streams need 8 lanes on both sides (otpu_dma W = 8: D >= 32). Without it DSTEP and
  // STREAM are illegal instructions (otpu_seq STREAMS: the slice stops with ERROR)
  localparam bit HAS_SS = HAS_DSTEP && LANES == 8 && D >= 32;
  // SE_R > 0: the streams on otpu_se_eng beside the VPU (the VPU keeps running VOPs during a
  // stream; SE_R rows of the state a cycle: otpu_dma SR), the VPU's engine never requested (its
  // tail still runs the composites). Timing only: every result is the VPU engine's bit for bit
  localparam bit SE_OWN = HAS_SS && SE_R > 0;
  localparam int SR = SE_OWN ? SE_R : 1;
  localparam int P_MXU = 0, P_DMA = 1, P_Q = 2, P_VA = 3, P_VB = 4, P_Q3 = 5, P_Q2 = 6, P_COLL = 7, NRP = 8;
  // the board build (TMEM replicated per read port, RPB >= NRP * LANES: reads never conflict;
  // WPB = 1, or the pumped one copy's 2): the arbiter only masks write banks (see the arbiter),
  // and the VPU's grant only takes its buffered writes (otpu_vpu WBUF)
  localparam bit ARB_MASK = (TMEM_IMPL == 0 || (TMEM_IMPL == 3 && TMEM_RPL)) &&
                            (RPB >= NRP * LANES) && (WPB == 1 || (TMEM_IMPL == 3 && WPB == 2));
  // TMEM_IMPL 1 (banked TMEM): three copies, each bank one read per cycle; the ports of one
  // group on different copies (the quantizer's three, the VPU's two), so a group alone always
  // fits: copy 0 MXU, QUANT, VPU A; copy 1 VPU B, QUANT 2; copy 2 DMA, QUANT 3, COLL. The arbiter
  // counts one read per port and bank (its lanes there read one word) per copy (gnt_k).
  // TMEM_NCP 4 (two pumped copies' four read slots): MXU and VPU A, VPU B and QUANT 3, DMA and
  // QUANT 2, QUANT and COLL
  // TMEM_NCP 2 (one pumped copy's two read slots, P1): MXU, VPU A, QUANT 2 and COLL share one,
  // DMA, VPU B and QUANT the other; QUANT 3 (the row factor, one word) has a third of its own, a
  // model of the row factor kept in a register by the quantizer (two slots cannot hold the
  // quantizer's three reads)
  // TMEM_NCP 5 / 7 (one copy pumped at 3x / 4x: 2K - 1 read slots, their count only, at today's
  // read latency): 5: MXU; DMA, QUANT 3 and COLL; QUANT; VPU A; VPU B and QUANT 2. 7: a slot per
  // port, COLL on the DMA's
  localparam int NCP = (TMEM_NCP == 2) ? 3 : TMEM_NCP;
  localparam logic [NRP-1:0][2:0] CPY =
    (TMEM_NCP == 2) ? {3'd0, 3'd0, 3'd2, 3'd1, 3'd0, 3'd1, 3'd1, 3'd0} :
    (TMEM_NCP == 4) ? {3'd3, 3'd2, 3'd1, 3'd1, 3'd0, 3'd3, 3'd2, 3'd0} :
    (TMEM_NCP == 5) ? {3'd1, 3'd4, 3'd1, 3'd4, 3'd3, 3'd2, 3'd1, 3'd0} :
    (TMEM_NCP == 7) ? {3'd1, 3'd6, 3'd5, 3'd4, 3'd3, 3'd2, 3'd1, 3'd0} :
                      {3'd2, 3'd1, 3'd2, 3'd1, 3'd0, 3'd0, 3'd2, 3'd0};
  //                    COLL  Q2    Q3    VB    VA    Q     DMA   MXU
  // TMEM_RQ: each read port's rank in the arbiter's priority order (its group's: DMA, COLL, VPU,
  // MXU, QUANT), for the copies' request-priority bank addresses (otpu_tmem RQ)
  localparam logic [NRP-1:0][2:0] PRANK = {3'd1, 3'd4, 3'd4, 3'd2, 3'd2, 3'd4, 3'd0, 3'd3};
  //                                       COLL  Q2    Q3    VB    VA    Q     DMA   MXU
  initial if (TMEM_IMPL == 1 && !(TMEM_NCP inside {2, 3, 4, 5, 7}))
    $fatal(1, "otpu_slice: TMEM_NCP %0d (2, 3, 4, 5 or 7)", TMEM_NCP);
  initial if (TMEM_IMPL == 1 && (RPB != 1 || WPB != 1))
    $fatal(1, "otpu_slice: TMEM_IMPL 1 needs RPB = 1 and WPB = 1 (RPB %0d WPB %0d)", RPB, WPB);
`ifdef SYNTHESIS
  initial if (TMEM_IMPL == 2) $fatal(1, "otpu_slice: TMEM_IMPL 2 is a simulation model");
  initial if (WG > 1 && (WG != NCH || D != CB))
    $fatal(1, "otpu_slice: WG %0d x D %0d is not port W's NCH %0d x CB %0d (a column's part a lane)",
           WG, D, NCH, CB);
`endif
  initial if (TMEM_IMPL == 2 && WPB != 1)
    $fatal(1, "otpu_slice: TMEM_IMPL 2 needs WPB = 1 (WPB %0d)", WPB);
  localparam int W_DMA = 0, W_MXU = 1, W_COLL = 2, W_VPU = 3, NWP = 4;
  // The VPU ahead of the MXU drain: the VPU writes all its lanes' banks at once, so an MXU drain
  // write in any one bank cost it the whole cycle, and decode's softmax (EXP2SUB, which the next
  // PV MM waits for) ran at about half its rate; the drain holds a cycle instead
  // (docs/litedram.md section 11, "The core's own gaps": Qwen3 4-bit -3.0% cycles per token).
  localparam int G_DMA = 0, G_COLL = 1, G_VPU = 2, G_MXU = 3, G_Q = 4, NG = 5;
  // TMEM_IMPL 3: every read port its own rank (the groups' order, DMA COLL VPU MXU QUANT; within
  // a group in port order) and its group
  localparam logic [NRP-1:0][2:0] PRANK3D = {3'd1, 3'd6, 3'd7, 3'd3, 3'd2, 3'd5, 3'd0, 3'd4};
  //                                        COLL  Q2    Q3    VB    VA    Q     DMA   MXU
  localparam logic [NRP-1:0][2:0] PRANK3 = (TMEM_PRK != 0) ? (3 * NRP)'(TMEM_PRK) : PRANK3D;
  localparam logic [NRP-1:0][2:0] RGRP = {3'(G_COLL), 3'(G_Q), 3'(G_Q), 3'(G_VPU), 3'(G_VPU),
                                          3'(G_Q), 3'(G_DMA), 3'(G_MXU)};
  // TMEM_TDP: the RAM port each read port asks on (1: port A). The VPU's and the quantizer's
  // operands never share a RAM port. Pumped (TMEM_PMP), where the write takes neither: QUANT and
  // VPU A on port A, the MXU, the DMA (with its guests), VPU B and QUANT 2 on port B (the fewest
  // replays of every split, docs/tmem_nocopy.md 6.4). Not pumped, where the writes take port A
  // first: VPU B, QUANT 3, QUANT 2 and COLL on port A, the busy readers on port B
  localparam logic [NRP-1:0] TM_SIDED = (TMEM_PMP != 0) ? ((NRP'(1) << P_Q) | (NRP'(1) << P_VA)) :
                                        ((NRP'(1) << P_VB) | (NRP'(1) << P_Q3) | (NRP'(1) << P_Q2) |
                                         (NRP'(1) << P_COLL));
  localparam logic [NRP-1:0] TM_SIDE = (TMEM_SDM >= 0) ? NRP'(TMEM_SDM) : TM_SIDED;
  // the MXU's TMEM interface (otpu_mxu TQ): with replays its reads are not granted (TMEM reads
  // what it asks and replays a lost read, as the VPU's), the grant takes its writes from a queue,
  // and its epilogue advances on flip-flops instead of the grant
  localparam bit MXU_TQ = TMEM_IMPL == 3 && TMEM_RPL;
  initial if (TMEM_IMPL == 3 && (!(WPB == 1 || (WPB == 2 && TMEM_PMP != 0 && TMEM_RPL)) ||
                                 ARB_MASK != TMEM_RPL))
    $fatal(1, "otpu_slice: TMEM_IMPL 3 needs WPB = 1 (WPB %0d; 2 pumped with TMEM_RPL), and RPB >= %0d with TMEM_RPL",
           WPB, NRP * LANES);

  // ---- sequencer
  cmd_t ucmd [NUNITS];
  logic [NUNITS-1:0] ustart, urdy, udone;
  // the sequencer's side of the units' handshake: the same, except in simulation with +tm_sdly
  // or +tm_ddly (docs/tmem_nocopy.md: the cycles a TMEM with a longer read path adds)
  cmd_t ucmd_s [NUNITS];
  logic [NUNITS-1:0] ustart_s, urdy_s, udone_s;
  logic urel;
  // FOLLOW (docs/isa.md "Wait counts"): wait entries across slices; the VPU's start follows a
  // running GATHER (ufol), the top of that GATHER's writes into this TMEM (f_top)
  localparam bit FOL = SEQ_WC && S > 1;
  logic ufol, ufol_s;
  logic [31:0] f_top;
  logic im_we;
  logic [31:0] im_row;
  logic sc_we;                    // SEQ_WC: the loader's sidecar row (wait entries)
  logic [31:0] sc_row;
  logic [255:0] sc_data;
  seq_ev_t sq_ev;
  logic        srst;              // the sequencer's and the units' reset: rst, a CHAIN reload, or
                                  //   a WAITW timeout (to_err)
  logic        sq_halted, sq_ch, sq_err, rl_v;
  logic        dma_err;           // a WAITW timed out: the slice stops with an error (to_err)
  logic [31:0] sq_icount, sq_cha, sq_chn;
  logic [AW-1:0] rl_val;
  // ICACHE: the program (pg_addr, pg_n; the cache emptied while ic_inv), the line fills' port B
  // reads (ic_req, ic_addr) and their chunks (ic_rv_i, a cycle after the memory's b_rvalid)
  logic        ic_inv, ic_req, ic_rv_i, ic_busy;
  logic [31:0] pg_addr, pg_n, ic_addr;
  otpu_seq #(.IMEM_WORDS(IMEM_WORDS), .TMEM_WORDS(TMEM_WORDS), .SID(SID), .S(S), .D(D),
             .WIN(WIN), .STREAMS(HAS_SS), .WG(WG), .WAITC(SEQ_WC), .STYLE(IMEM_STYLE),
             .WSTYLE(IMEM_WC_STYLE),
             .DSTEP2(DSTEP2 && HAS_SS), .FOLLOW(FOL), .QB(QB), .AW(AW),
             .ICACHE(ICACHE), .IC_LINES(IC_LINES), .IC_PF(IC_PF), .QONCE(QONCE)) u_seq (
    .clk, .rst(srst), .ucmd(ucmd_s), .ustart(ustart_s), .urel, .ufol(ufol_s), .urdy(urdy_s),
    .udone(udone_s),
    .halted(sq_halted), .error(sq_err),
    .icount(sq_icount), .ev(sq_ev), .rinit, .rl_v, .rl_val, .ch_req(sq_ch), .ch_addr(sq_cha),
    .ch_n(sq_chn), .im_we, .im_row, .im_data(b_rdata_i),
    .sc_we, .sc_row, .sc_data,
    .ic_inv, .ic_base(pg_addr), .ic_n(pg_n), .ic_req, .ic_addr, .ic_gnt(b_rdy_i), .ic_rv(ic_rv_i), .ic_busy);

  // The units' handshake. In simulation, +tm_sdly=N starts every DMA, QUANT, VPU and collective
  // instruction N cycles after the sequencer issues it (+tm_sidle=1: only when its unit has
  // nothing in flight, as if a busy unit's next reads were fetched ahead), and +tm_ddly=N
  // reports every completion N cycles late (its writes land later): the latency a TMEM with
  // stream buffers or a registered read path adds (docs/tmem_nocopy.md). The MXU starts as
  // issued (its stream and release, urel, are in order with its start). 0: wires.
`ifdef SYNTHESIS
  assign ustart = ustart_s;
  assign ufol = ufol_s;
  assign urdy_s = urdy;
  assign udone_s = udone;
  for (genvar u = 0; u < NUNITS; u++) begin : g_hs
    assign ucmd[u] = ucmd_s[u];
  end
`else
  int hs_sd, hs_dd, hs_idle;
  initial begin
    hs_sd = 0; hs_dd = 0; hs_idle = 0;
    void'($value$plusargs("tm_sdly=%d", hs_sd));
    void'($value$plusargs("tm_ddly=%d", hs_dd));
    void'($value$plusargs("tm_sidle=%d", hs_idle));
    if (hs_dd > 15) $fatal(1, "otpu_slice: +tm_ddly %0d (0..15)", hs_dd);
  end
  logic [NUNITS-1:0] hs_pend, hs_fire, hs_dly;
  int                hs_cnt [NUNITS], hs_busy [NUNITS];
  cmd_t              hs_cmd [NUNITS];
  logic              hs_fol;
  logic [NUNITS-1:0] hs_dq [16];
  always_comb
    for (int u = 0; u < NUNITS; u++) begin
      hs_dly[u] = hs_sd > 0 && u != U_MXU && (hs_idle == 0 || hs_busy[u] == 0);
      hs_fire[u] = hs_pend[u] && hs_cnt[u] == 1;
      ustart[u] = hs_fire[u] || (ustart_s[u] && !hs_dly[u]);
      ucmd[u] = hs_fire[u] ? hs_cmd[u] : ucmd_s[u];
      urdy_s[u] = urdy[u] && !hs_pend[u];
      udone_s[u] = (hs_dd == 0) ? udone[u] : hs_dq[hs_dd - 1][u];
    end
  assign ufol = hs_fire[U_VPU] ? hs_fol : ufol_s;
  always_ff @(posedge clk) begin
    if (srst) begin
      hs_pend <= '0;
      for (int u = 0; u < NUNITS; u++) begin hs_cnt[u] <= 0; hs_busy[u] <= 0; end
    end else
      for (int u = 0; u < NUNITS; u++) begin
        if (ustart_s[u] && hs_dly[u]) begin
          hs_pend[u] <= 1'b1; hs_cnt[u] <= hs_sd; hs_cmd[u] <= ucmd_s[u];
          if (u == U_VPU) hs_fol <= ufol_s;
        end else if (hs_pend[u]) begin
          hs_cnt[u] <= hs_cnt[u] - 1;
          if (hs_cnt[u] == 1) hs_pend[u] <= 1'b0;
        end
        hs_busy[u] <= hs_busy[u] + int'(ustart[u]) - int'(udone[u]);
      end
    hs_dq[0] <= srst ? '0 : udone;
    for (int i = 1; i < 16; i++) hs_dq[i] <= srst ? '0 : hs_dq[i - 1];
  end
`endif

  // ---- HALT CHAIN: reload IMEM from the chained program's address, the units held in reset.
  // A WAITW timeout (dma_err) stops the slice until rst: to_err holds the sequencer and the
  // units in reset, so no instruction starts or ends after it (TMEM, DRAM and ICOUNT stay as
  // they were), and HALTED rises once they are in reset (to_q: the units' resets are up to two
  // registers behind srst) and the DRAM has taken every write they gave it (wr_idle). ICOUNT
  // keeps the sequencer's count as it stood (ic_base, as at a CHAIN reload)
  logic        c_rst, c_ld, to_err;
  logic [2:0]  to_q;
  logic [31:0] c_addr, c_n, ic_base;
  assign srst = rst || c_rst || to_err;
  assign halted = (sq_halted && !sq_ch) || (to_q[2] && wr_idle_i);
  assign error = sq_err || to_err;
  assign wait_to = to_err;
  assign icount = ic_base + sq_icount;
  always_ff @(posedge clk) begin
    c_ld <= 1'b0;
    to_q <= rst ? 3'b000 : {to_q[1:0], to_err};
    if (rst) begin
      c_rst <= 1'b0; ic_base <= '0; to_err <= 1'b0;
    end else if (dma_err || to_err) begin
      to_err <= 1'b1;
      if (to_err && !to_q[0]) ic_base <= ic_base + sq_icount;   // its last cycle out of reset
    end else if (!c_rst) begin
      if (sq_halted && sq_ch && wr_idle_i) begin
        c_rst <= 1'b1; c_ld <= 1'b1;
        c_addr <= sq_cha; c_n <= sq_chn;
        ic_base <= ic_base + sq_icount;
      end
    end else if (!c_ld && !ld_busy) c_rst <= 1'b0;      // loaded (the loader starts after c_ld)
  end

  // ---- program loader (see the top): ld_go once the DRAM had no read in flight; ld_ch: a HALT
  // CHAIN reload (rst stops it). While busy the loader has port B to itself.
  // SEQ_WC (docs/isa.md "Wait counts"): the program's wait entries follow its instructions in
  // DRAM (from the next chunk on, 64 bits per instruction), and the loader reads them into the
  // sequencer's sidecar RAM after the instructions, 4 entries a row: each chunk is read D / 32
  // times, each read writing one row (its 256-bit part), so no chunk is held
  // ICACHE: no copy: the loader waits as before (and for the cache's fills), then empties the
  // cache with the program's address and length (ic_inv, the cycle it ends)
  localparam int IPR = D / 32;
  logic [31:0] ld_rows, ld_tot, ld_iss, ld_cmp, ld_a;
  logic        ld_go, ld_ch;
  wire ld_req = !ICACHE && ld_busy && ld_go && ld_iss < ld_tot;
  assign ic_inv = ICACHE && (sys_rst || (ld_busy && ld_go));
  wire [31:0] ld_n_s = ld_start ? ld_n : c_n;
  wire        ld_wt = SEQ_WC && ld_cmp >= ld_rows;     // this read is a row of wait words
  wire [31:0] ld_sr = ld_cmp - ld_rows;                 // its sidecar row
  assign im_we  = !ICACHE && ld_busy && ld_go && b_rvalid_i && !ld_wt;
  assign im_row = ld_cmp;
  assign sc_we  = !ICACHE && ld_busy && ld_go && b_rvalid_i && ld_wt;
  assign sc_row = ld_sr;
  assign sc_data = b_rdata_i[256 * (ld_sr % IPR) +: 256];
  always_ff @(posedge clk) begin
    if (sys_rst) begin
      ld_busy <= 1'b0;
    end else if (ld_start || (c_ld && !ld_busy)) begin
      ld_rows <= (ld_n_s + IPR - 1) / IPR;
      ld_tot <= (ld_n_s + IPR - 1) / IPR + (SEQ_WC ? (ld_n_s + 3) / 4 : 0);
      ld_iss <= '0; ld_cmp <= '0;
      ld_a <= (ld_start ? ld_addr : c_addr) >> 2;
      ld_busy <= (ld_n_s != 0);
      ld_go <= 1'b0;
      ld_ch <= !ld_start;
      pg_addr <= ld_start ? ld_addr : c_addr;
      pg_n <= ld_n_s;
    end else if (ld_busy) begin
      if (rst && ld_ch) ld_busy <= 1'b0;
      else if (!ld_go) ld_go <= rd_idle_i && !ic_busy;
      else if (ICACHE) ld_busy <= 1'b0;
      else begin
        if (ld_req && b_rdy_i) begin
          ld_iss <= ld_iss + 1;
          // the next chunk; a chunk of wait words after its D / 32 reads
          if (ld_iss < ld_rows || (ld_iss - ld_rows) % IPR == IPR - 1) ld_a <= ld_a + D / 4;
        end
        if (b_rvalid_i) begin
          ld_cmp <= ld_cmp + 1;
          if (ld_cmp + 1 == ld_tot) ld_busy <= 1'b0;
        end
      end
    end
  end

  // ---- TMEM
  logic [NRP-1:0][LANES-1:0]       rq_en, r_en;
  logic [NRP-1:0]                  tm_bx, tm_bs, tm_sv, tm_pre;   // TMEM_IMPL 3 (below)
  logic [NRP-1:0]                  tm_lost, tm_frz;               // TMEM_IMPL 3, TMEM_RPL
  logic [NRP-1:0][LANES-1:0]       rq_t;
  logic [NRP-1:0][LANES-1:0][31:0] r_addr, r_data;
  // the units' reads by bank (otpu_tmem PBR, from their flip-flops: the MXU's under MXU_TQ, the
  // DMA's, the quantizer's and the VPU's with the one TMEM copy), the banks held as rq_t is
  localparam logic [NRP-1:0] PBR = (TMEM_IMPL == 3) ? ((MXU_TQ ? NRP'(1) << P_MXU : '0) |
      (NRP'(1) << P_DMA) | (NRP'(1) << P_Q) | (NRP'(1) << P_Q2) | (NRP'(1) << P_VA) |
      (NRP'(1) << P_VB)) : '0;
  logic [NRP-1:0][LANES-1:0]       rq_b, r_ben;
  logic [NRP-1:0][LANES-1:0][31:0] r_bad;
  assign rq_b[P_Q3] = '0, r_bad[P_Q3] = '0, rq_b[P_COLL] = '0, r_bad[P_COLL] = '0;   // (no PBR)
  logic [NWP-1:0][LANES-1:0]       wq_en, w_en;
  logic [NWP-1:0][LANES-1:0][31:0] w_addr, w_data;
  logic [NWP-1:0]                  w_gnt;   // each write port's grant (the DMA always writes)
  // the MXU's ports keep the crossbar (its drain writes scattered rows); the others are rotators.
  // WG > 1 (GALL): every MXU write is a run from lane 0 (a result row's consecutive words, pg's
  // halves, RMAX's mxo + mx_i + k), so its write port is a rotator too (yosys otpu_tmem at
  // bcu1525-x2:wmx's parameters -1,452 LUT); its read port keeps the crossbar (PBR)
  // The quantizer's row-factor port (one word) and the collective's read port, rarely busy,
  // share the DMA's copy (ST reads): 6 copies of the memory instead of 8.
  // (TMEM_IMPL 2, simulation: IMPL 0 with a copy per read port, so any grant reads the right
  // words; otpu_tmem_model decides the grants). IQ: TMEM's input stage (release r2,
  // docs/tmem_nocopy.md 7): the granted writes registered in TMEM, the units' read requests in
  // the units (each a cycle after it forms), so both move a cycle and the order holds.
  // rst_mem: the memories' control state (TMEM's write valids, replays and their ages, the
  // pumped RAMs' phase; the ACT RAM's write enables) resets with the slice's power-on reset, not
  // the run's (a run's reset leaves the memories' writes landing); its own register, replicated
  // by its loads (as the units' below).
  // OTPU_FPGA_INIT (the FPGA flows' create_project*.tcl): no such reset. Configuration loads every
  // flip-flop's INIT 0, the reset's state (the pumped RAMs' tq, reset 1, samples tog and aligns
  // at the first fast edges); afterwards that state is a pipeline of the units' requests, and
  // every reset of the units (a run's, and sys_rst's: core_rst follows it) drains it with no
  // request coming, as a run's reset always has (docs/asic.md "Power-up state"). The ASIC and
  // the simulations keep it: their flip-flops start unknown
  (* max_fanout = 64 *) logic rst_mem;
`ifdef OTPU_FPGA_INIT
  assign rst_mem = 1'b0;
`else
  always_ff @(posedge clk) rst_mem <= sys_rst;
`endif
  otpu_tmem #(.WORDS(TMEM_WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(WPB),
              .GEN_R(NRP'(1) << P_MXU), .GEN_W((WG > 1) ? NWP'(0) : NWP'(1) << W_MXU),
              .PBR(PBR),
              .SH_HOST(P_DMA),
              .SH_MASK((TMEM_IMPL == 2) ? '0 : (NRP'(1) << P_Q3) | (NRP'(1) << P_COLL)),
              .SID(SID), .IMPL((TMEM_IMPL == 2) ? 0 : TMEM_IMPL), .NCP(NCP), .CPY(CPY), .RQ(TMEM_RQ),
              .PRANK((TMEM_IMPL == 3) ? PRANK3 : PRANK), .NSB(TMEM_NSB), .SBP(TMEM_SBP),
              .GRP(RGRP), .SBH(TMEM_SBH), .RPL(TMEM_RPL), .RPLM(~(NRP'(1) << P_COLL)),
              .TOPP(P_DMA), .TDP(TMEM_TDP), .SIDE(TM_SIDE), .GST(TMEM_GST),
              .PMP(TMEM_PMP), .IQ(1'b1), .STYLE(TMEM_STYLE)) u_tmem (
    .clk, .clk_f, .rst(rst_mem), .r_en, .r_req(rq_t), .r_addr, .r_ben, .r_bad, .r_data, .w_en,
    .w_req(wq_en), .w_gnt, .w_addr, .w_data, .blk_x(tm_bx), .blk_s(tm_bs), .lost(tm_lost),
    .frz(tm_frz), .dump);
  assign coll_rdata = r_data[P_COLL];
  // the units' TMEM requests
  logic [LANES-1:0]        dma_ren, dma_wen, va_ren, vb_ren, v_wen;
  logic                    v_ren;   // the VPU's lanes take their read data this cycle
  logic [LANES-1:0][31:0]  dma_raddr, dma_waddr, dma_wdata, va_raddr, vb_raddr, v_waddr, v_wdata;
  logic [ULANES-1:0]       mxu_ren, mxu_wen, q_ren, q_ren2;
  logic [ULANES-1:0][31:0] mxu_raddr, mxu_waddr, mxu_wdata, q_raddr, q_raddr2;

  // ---- ACT RAM
  logic [ULANES*QUANT_PW-1:0]      act_we;     // (the quantizer's writer: ULANES x QUANT_PW lanes)
  logic [ULANES*QUANT_PW-1:0][7:0] act_data;
  logic                   asc_we;
  logic [7:0]             act_row, asc_row, act_off;
  logic                   act_dup;
  logic [31:0]            act_idx;
  logic [63:0]            asc_data;     // {da3, da2, da1, da0, Sa}
  logic [15:0]            asc_blk, act_rblk, act_rblk2;
  logic [MCOLS-1:0]       act_rhi;
  logic [7:0]             act_rgrp;
  logic [MCOLS-1:0]       act_ren;       // per column (otpu_mxu: the MM's columns)
  logic                   mxu_pop;       // the MXU consumes a block this cycle (its ACT read)
  logic [MCOLS*D*8-1:0]   act_rdata;
  logic [MCOLS*64-1:0]    act_rscale;
  // QONCE: the DMA's LDA writes (otpu_actram DPORT), taken while the quantizer has no write at its
  // write stage (q_actwv)
  logic [ULANES*QUANT_PW-1:0]      la_we;
  logic [ULANES*QUANT_PW-1:0][7:0] la_data;
  logic [7:0]             la_row, la_srow;
  logic [31:0]            la_idx;
  logic                   la_swe, q_actwv;
  logic [15:0]            la_sblk;
  logic [63:0]            la_sdata;
  // AASK: the ACT RAM reads by the MXU's chain stages (otpu_actram ASK: its CLS)
  localparam int ACT_ASK = (MXU_AASK && WG > 1 && MXU_PUMP == 0) ? ((D / 8 < MXU_CL) ? D / 8 : MXU_CL)
                                                                  : 0;   // (PUMP: no AASK)
  otpu_actram #(.D(D), .MCOLS(MCOLS), .ROWS(ACT_ROWS), .BLOCKS(ACT_BLOCKS), .LANES(ULANES * QUANT_PW),
                .UCOLS(ACT_UCOLS), .ASK(ACT_ASK), .DPORT(QONCE)) u_act (
    .clk, .rst(rst_mem), .we(act_we), .w_row(act_row), .w_idx(act_idx), .w_data(act_data),
    .w_dup(act_dup), .w_off(act_off), .swe(asc_we),
    .s_row(asc_row), .s_blk(asc_blk), .s_data(asc_data),
    .d_we(la_we), .d_row(la_row), .d_idx(la_idx), .d_data(la_data), .d_swe(la_swe),
    .d_srow(la_srow), .d_sblk(la_sblk), .d_sdata(la_sdata), .ren(act_ren), .r_use(mxu_pop), .r_blk(act_rblk),
    .r_blk2(act_rblk2), .r_hi(act_rhi), .r_grp(act_rgrp),
    .r_data(act_rdata), .r_scale(act_rscale));

  // ---- units
  logic [NG-1:0] gnt;
  logic        q3_en;
  logic [31:0] q3_addr;
  logic d_dma, d_mxu, d_q, d_vpu, r_dma, r_mxu, r_q, r_vpu;
  logic mxu_areq, q_areq, q_awant, q_awe, mxu_agnt, mxu_bgnt;
  logic q_ahold;                            // QONCE: the quantizer's writer held for a record
  // the quantizer's grant waits while the SW port cannot take its write, or for a_hold
  wire  q_blk = (q_awant && !sw_rdy_i) || q_ahold;
  logic [WA-1:0] mxu_aaddr, q_aaddr;
  logic [32*QUANT_SWW-1:0] q_awdata;
  logic [4*QUANT_SWW-1:0] q_abe;
  logic dma_breq, dma_bwe, mxu_breq;
  logic [D/4-1:0] dma_bwmask;
  logic [D*8-1:0] dma_bwdata;
  logic [WA-1:0] dma_baddr, mxu_baddr;
  logic        mxu_pwreq;                   // the MXU's port W (WG > 1)
  logic [WA-1:0] mxu_pwaddr;
  localparam int FW = $clog2(FIFO_DEPTH) + 1;
  logic [FW-1:0] mxu_level;
  logic mxu_starve, mxu_block, mxu_u, q_u, v_u;
  logic [3:0][31:0] mxu_uv;
  logic [31:0] q_frz, v_frz;

  // Each unit's reset is its own register (a reset tree): the slice's reset reaches ~15k flip-
  // flops across the die, and one replicated net from the board ran 8.4 ns routes into the
  // units (the worst core_clk paths at 114 MHz). The units leave reset a cycle after the
  // sequencer, which starts nothing that early. max_fanout 64: at 125.49 MHz rst_vpu (256 per
  // copy) into the VPU lanes' c registers had 0.25 ns slack. Q, the VPU and the MXU take a
  // second stage (rst_q0, rst_v0, rst_m0), so their replicas are loaded next to them: in the SE
  // v2 build the board's core_rst copies into rst_vpu / rst_q had -0.107 ns with no logic, on
  // 812bb01 rst_mxu into the MXU's weight FIFO pointers +0.123 ns with none. They leave reset
  // two cycles after the sequencer (checked below).
  (* max_fanout = 64 *) logic rst_dma, rst_mxu, rst_q, rst_vpu;
  logic rst_q0, rst_v0, rst_m0;
  always_ff @(posedge clk) begin
    rst_dma <= srst; rst_m0 <= srst; rst_q0 <= srst; rst_v0 <= srst;
    rst_mxu <= rst_m0; rst_q <= rst_q0; rst_vpu <= rst_v0;
  end
`ifndef SYNTHESIS
  // the check starts once the reset has held two edges (registers start arbitrary, the units'
  // reset registers above among them)
  bit rst_p, rst_seen;
  initial begin rst_p = 1'b0; rst_seen = 1'b0; end
  always @(posedge clk) begin rst_p <= srst; if (srst && rst_p) rst_seen <= 1'b1; end
  always_ff @(posedge clk)
    if (rst_seen && ((ustart[U_DMA] && rst_dma) || (ustart[U_MXU] && rst_mxu) || (ustart[U_Q] && rst_q) ||
                     (ustart[U_VPU] && rst_vpu)))
      $fatal(1, "otpu_slice: a unit started while in reset (ustart %b)", ustart);
`endif

  // the stream engine (HAS_SS, SE_OWN above): the DMA's side (ss_*) and the VPU's (vs_*: the
  // same nets, or with SE_OWN tied off)
  logic        ss_req, ss_gnt, ss_pe, ss_in_v, ss_y_v, ss_o_v;
  ss_cfg_t     ss_cfg;
  logic [2:0]  ss_fk;
  logic [4:0]  ss_fi;
  logic [31:0] ss_in_d [SR*LANES], ss_fd [LANES], ss_y_d [SR*LANES];
  logic [SR-1:0][31:0] ss_o_d;
  logic        vs_req, vs_gnt, vs_pe, vs_in_v, vs_y_v, vs_o_v;
  logic [2:0]  vs_fk;
  logic [31:0] vs_in_d [LANES], vs_y_d [LANES];
  logic [31:0] vs_o_d;
  if (SE_OWN) begin : g_se
    otpu_se_eng #(.LANES(LANES), .R(SE_R), .DSTEP2(DSTEP2)) u_se (
      .clk, .rst(rst_dma), .ss_req, .ss_gnt, .ss_cfg, .ss_pe, .ss_in_v, .ss_in_d, .ss_fk, .ss_fi,
      .ss_fd, .ss_y_v, .ss_y_d, .ss_o_v, .ss_o_d);
    assign vs_req = 1'b0;
    assign vs_pe = 1'b0;
    assign vs_in_v = 1'b0;
    assign vs_fk = SF_NONE;
    for (genvar l = 0; l < LANES; l++) begin : g_l
      assign vs_in_d[l] = '0;
    end
  end else begin : g_vse
    assign vs_req = ss_req;
    assign ss_gnt = vs_gnt;
    assign vs_pe = ss_pe;
    assign vs_in_v = ss_in_v;
    assign vs_fk = ss_fk;
    assign vs_in_d = ss_in_d;
    assign ss_y_v = vs_y_v;
    assign ss_y_d = vs_y_d;
    assign ss_o_v = vs_o_v;
    assign ss_o_d = vs_o_d;
  end

  otpu_dma #(.D(D), .LANES(LANES), .DEPTH(DMA_DEPTH), .LDQ(DMA_LDQ), .HAS_DSTEP(HAS_SS), .SR(SR),
             .DSTEP2(DSTEP2 && HAS_SS), .ST16(HAS_SS ? ST16 : 0), .ST16S(ST16S), .AW(AW),
             .LDA(QONCE), .ALANES(ULANES * QUANT_PW)) u_dma (
    .clk, .rst(rst_dma), .start(ustart[U_DMA]), .cmd(ucmd[U_DMA]), .rdy(r_dma), .done(d_dma),
    .err(dma_err), .ww_held(a_inval),
    .b_req(dma_breq), .b_gnt(b_rdy_i && !ic_req), .b_we(dma_bwe), .b_wmask(dma_bwmask),
    .b_wdata(dma_bwdata), .b_addr(dma_baddr), .b_rvalid(b_rvalid_i && b_rtag_i && !ic_rv_i),
    .b_rdata(b_rdata_i), .wr_idle(wr_idle_i),
    .t_ren(dma_ren), .t_raddr(dma_raddr), .t_rben(rq_b[P_DMA]), .t_rbad(r_bad[P_DMA]),
    .t_rdata(r_data[P_DMA]),
    .t_wen(dma_wen), .t_waddr(dma_waddr), .t_wdata(dma_wdata),
    .ss_req, .ss_gnt, .ss_cfg, .ss_pe, .ss_in_v, .ss_in_d, .ss_fk, .ss_fi, .ss_fd,
    .ss_y_v, .ss_y_d, .ss_o_v, .ss_o_d,
    .act_busy(q_actwv), .act_we(la_we), .act_row(la_row), .act_idx(la_idx), .act_data(la_data),
    .asc_we(la_swe), .asc_row(la_srow), .asc_blk(la_sblk), .asc_data(la_sdata));

  otpu_mxu #(.D(D), .MCOLS(MCOLS), .ROWS(ACT_ROWS), .DEPTH(FIFO_DEPTH), .LANES(ULANES), .IMPL(MXU_IMPL),
             .CL(MXU_CL), .E(MXU_E), .QD(MXU_QD), .WG(WG), .AB((WG > 1) ? CB : 32), .TQ(MXU_TQ), .SID(SID),
             .RF_STYLE(MXU_RF_STYLE), .FIFO_STYLE(MXU_FIFO_STYLE), .WASK(MXU_WASK),
             .RF_ROWS(MXU_RF_ROWS), .BRAW(MXU_BRAW), .AASK(MXU_AASK), .PUMP(MXU_PUMP != 0),
             .XPOST(XPOST), .QB(XQB),
             .CHASH(CHASH), .KT(KT), .SPL(MXU_SPL != 0),
             .TLANES(LANES), .DAW(AW)) u_mxu (
    .clk, .clk_f, .rst(rst_mxu), .start(ustart[U_MXU]), .go(urel), .cmd(ucmd[U_MXU]), .rdy(r_mxu), .done(d_mxu),
    .computing(mxu_pop), .pf_level(mxu_level), .pf_starve(mxu_starve), .pf_block(mxu_block),
    .pf_u(mxu_u), .pf_uv(mxu_uv),
    .act_blk(act_rblk), .act_blk2(act_rblk2), .act_hi(act_rhi), .act_grp(act_rgrp), .act_ren, .act_data(act_rdata), .act_scale(act_rscale),
    .a_req(mxu_areq), .a_addr(mxu_aaddr), .a_gnt(mxu_agnt), .a_rvalid, .a_rdata,
    .b_req(mxu_breq), .b_addr(mxu_baddr), .b_gnt(mxu_bgnt), .b_rvalid(b_rvalid_i && !b_rtag_i),
    .b_rdata(b_rdata_i),
    .w_req(mxu_pwreq), .w_addr(mxu_pwaddr), .w_gnt(WG > 1 && pw_rdy),
    .w_bv((WG > 1) ? pw_bv[WG-1:0] : '0),
    .w_bd(pw_bd[WG*D*8-1:0]),
    .w_slot(pw_slot), .w_one(pw_one), .p_clk, .p_we(p_we[WG-1:0]), .p_q(p_q[WG-1:0]),
    .p_slot(p_slot[WG-1:0]), .p_d(p_d[WG-1:0]), .p_cnt(p_cnt[WG-1:0]),
    .t_ren(mxu_ren), .t_raddr(mxu_raddr), .t_rdata(r_data[P_MXU][ULANES-1:0]),
    .t_wen(mxu_wen), .t_waddr(mxu_waddr), .t_wdata(mxu_wdata), .t_gnt(gnt[G_MXU]),
    .t_lost(tm_lost[P_MXU]), .t_rben(rq_b[P_MXU]), .t_rbad(r_bad[P_MXU]));

  // the quantizer's group maxes of a row (MX offsets; not ROW mode): a QACT's 4 KB (KB <=
  // ACT_BLOCKS: the ACT RAM's bound) and a QST's KB (its row in TMEM: KB <= TMEM_WORDS / D), so
  // every QACT and QST the ISA's bounds allow fits (the card: 512)
  localparam int QG = (4 * ACT_BLOCKS > TMEM_WORDS / D) ? 4 * ACT_BLOCKS : TMEM_WORDS / D;
  localparam int Q_GR = (QG <= 8) ? 8 : (1 << $clog2(QG));
  otpu_quant #(.D(D), .LANES(ULANES), .TLANES(LANES), .GR(Q_GR), .XD(QUANT_XD), .R1(QUANT_R1),
               .PW(QUANT_PW), .SPLIT(QUANT_SPLIT), .QHL(QUANT_HL), .SWW(QUANT_SWW), .QSTB(QUANT_QSTB), .QSTHL(QUANT_QSTHL), .QONCE(QONCE), .WG(WG), .CHASH(CHASH), .SID(SID), .AW(AW)) u_quant (
    .clk, .rst(rst_q), .start(ustart[U_Q]), .cmd(ucmd[U_Q]), .rdy(r_q), .done(d_q), .gnt(gnt[G_Q]),
    .t_ren(q_ren), .t_raddr(q_raddr), .t_rdata(r_data[P_Q][ULANES-1:0]),
    .t_ren2(q_ren2), .t_raddr2(q_raddr2), .t_rdata2(r_data[P_Q2][ULANES-1:0]),
    .t_rben(rq_b[P_Q]), .t_rbad(r_bad[P_Q]), .t_rben2(rq_b[P_Q2]), .t_rbad2(r_bad[P_Q2]),
    .t_ren3(q3_en), .t_raddr3(q3_addr), .t_rdata3(r_data[P_Q3][0]),
    .act_we, .act_row, .act_idx, .act_data, .act_dup, .act_off, .asc_we, .asc_row, .asc_blk, .asc_data,
    .a_want(q_awant), .a_hold(q_ahold), .a_rdy(sw_rdy_i), .act_wv(q_actwv), .wr_idle(wr_idle_i),
    .a_req(q_areq), .a_we(q_awe), .a_addr(q_aaddr), .a_wdata(q_awdata), .a_be(q_abe),
    .pf_u(q_u), .pf_frz(q_frz));

  otpu_vpu #(.LANES(LANES), .CL(VPU_CL), .SID(SID), .WBUF(ARB_MASK), .HAS_SE(HAS_SS),
             .SE_SS(!SE_OWN), .OUTER(OUTER), .DSTEP2(DSTEP2), .FOLLOW(FOL), .CNS(VPU_CNS),
             .FSUM(VPU_FSUM != 0 && HAS_SS), .VSIG(VPU_SIG != 0 && HAS_SS),
             .MAC(VPU_MAC != 0), .OUTER1(VPU_OUTER1 != 0)) u_vpu (
    .clk, .rst(rst_vpu), .start(ustart[U_VPU]), .cmd(ucmd[U_VPU]), .rdy(r_vpu), .done(d_vpu),
    .fol(ufol), .fol_top(f_top), .fol_done(coll_ack),
    .gnt(gnt[G_VPU]), .ren(v_ren), .rhold(tm_lost[P_VA] || tm_lost[P_VB]),
    .ta_en(va_ren), .ta_addr(va_raddr), .ta_data(r_data[P_VA]),
    .tb_en(vb_ren), .tb_addr(vb_raddr), .tb_data(r_data[P_VB]),
    .ta_ben(rq_b[P_VA]), .ta_bad(r_bad[P_VA]), .tb_ben(rq_b[P_VB]), .tb_bad(r_bad[P_VB]),
    .tw_en(v_wen), .tw_addr(v_waddr), .tw_data(v_wdata), .pf_u(v_u), .pf_frz(v_frz),
    .ss_req(vs_req), .ss_gnt(vs_gnt), .ss_cfg, .ss_pe(vs_pe), .ss_in_v(vs_in_v),
    .ss_in_d(vs_in_d), .ss_fk(vs_fk), .ss_fi, .ss_fd, .ss_y_v(vs_y_v), .ss_y_d(vs_y_d),
    .ss_o_v(vs_o_v), .ss_o_d(vs_o_d));

  // FOL: the top of the running GATHER's writes into this TMEM: its dst at the sequencer's start
  // (before any of its writes lands, after every write of the collective before it), then each
  // granted collective write's end (a GATHER a VOP follows writes one run upwards from dst, its
  // lanes low first); across cards only local writes advance it (coll_wtop), so phase 1
  // followers must read the local half only. Only a follower
  // reads it (otpu_vpu FOLLOW)
  if (FOL) begin : g_ftop
    always_ff @(posedge clk) begin
      logic [$clog2(LANES + 1)-1:0] n;
      n = '0;
      for (int l = 0; l < LANES; l++) n = n + $bits(n)'(coll_wen[l]);
      if (ustart_s[U_COLL] && ucmd_s[U_COLL].op == OP_GATHER) f_top <= ucmd_s[U_COLL].w2;
      else if (|coll_wen && coll_gnt && coll_wtop) f_top <= coll_waddr[0] + 32'(n);
    end
  end else begin : g_nftop
    assign f_top = '0;
  end

  // collective: request from start until acknowledged (an RLD is not a collective)
  wire rl_start = ustart[U_COLL] && ucmd[U_COLL].op == OP_RLD;
  always_ff @(posedge clk) begin
    if (srst) coll_req <= 1'b0;
    else if (ustart[U_COLL] && !rl_start) coll_req <= 1'b1;
    else if (coll_ack) coll_req <= 1'b0;
  end
  always_comb begin                   // (no DRAM address there: w1h .. w3h 0)
    coll_cmd = ucmd[U_COLL];
    coll_cmd.w1h = '0; coll_cmd.w2h = '0; coll_cmd.w3h = '0;
  end
  // RLD: the word through the collective's read port (lane 0) once granted; the lane has it
  // the next cycle (rl_got), registered (rl_cv), converted (rl_c1), times the multiplier
  // (1 without MUL; AW bits, {w2h, w2}) as partial products (rl_c2: the low AW bits of
  // x * m[15:0], the low AW - 16 of x * m[31:16] and, at AW > 32, the low AW - 32 of
  // x * m[AW-1:32]), summed mod 2^AW and handed to the sequencer (rl_v). Without MUL the
  // multiplier is 1 and the sum is the converted word: rl_val takes it as rl_x does, from rl_w
  // at rl_cv, and rl_v follows, two cycles sooner; rl_c1 and rl_c2 run for MUL alone (an RLD's
  // latency is the bucketing's chain, docs/pf500.md). The read is
  // asked from rl_pend's second cycle (rl_pq): TMEM's input stage puts every unit's read a cycle
  // after it forms, and the writes a cycle after their grant, so RLD's read moves with them
  // (release r2, docs/tmem_nocopy.md 7)
  logic        rl_pend, rl_pq, rl_got, rl_cv, rl_c1, rl_c2, rl_raw, rl_mul;
  logic [31:0] rl_a, rl_w, rl_x;
  logic [AW-1:0] rl_m, rl_pl, rl_pt;
  logic [AW-17:0] rl_ph;
  wire         rl_busy = rl_pend || rl_got || rl_cv || rl_c1 || rl_c2 || rl_v;
  wire [31:0]  rl_xc = rl_raw ? rl_w : rld_f2i(rl_w);   // the word converted, valid at rl_cv
  always_ff @(posedge clk) begin
    if (srst) begin
      rl_pend <= 1'b0; rl_pq <= 1'b0; rl_got <= 1'b0; rl_cv <= 1'b0; rl_c1 <= 1'b0; rl_c2 <= 1'b0;
      rl_v <= 1'b0;
    end else begin
      rl_got <= rl_pq && r_en[P_COLL][0];
      rl_pq <= rl_pend && !(rl_pq && r_en[P_COLL][0]);
      if (rl_start) begin
        rl_pend <= 1'b1;
        rl_a <= ucmd[U_COLL].w1;
        rl_raw <= ucmd[U_COLL].flags[RF_RAW];
        rl_mul <= ucmd[U_COLL].flags[RF_MUL];
        rl_m <= ucmd[U_COLL].flags[RF_MUL] ? AW'(cmd_a(ucmd[U_COLL], 2)) : AW'(1);
      end else if (rl_pq && r_en[P_COLL][0]) rl_pend <= 1'b0;
      rl_cv <= rl_got;
      if (rl_got) rl_w <= r_data[P_COLL][0];
      rl_c1 <= rl_cv && rl_mul;
      if (rl_cv) rl_x <= rl_xc;
      rl_c2 <= rl_c1;
      if (rl_c1) begin
        rl_pl <= rl_x * rl_m[15:0];
        rl_ph <= rl_x[AW-17:0] * rl_m[31:16];
      end
      rl_v <= rl_c2 || rl_cv && !rl_mul;
      if (rl_c2) rl_val <= (AW > 32) ? rl_pl + {rl_ph, 16'd0} + rl_pt : rl_pl + {rl_ph, 16'd0};
      else if (rl_cv && !rl_mul) rl_val <= AW'(rl_xc);
    end
  end
  if (AW > 32) begin : g_rlt
    logic [AW-33:0] t;
    always_ff @(posedge clk) if (rl_c1) t <= rl_x[AW-33:0] * rl_m[AW-1:32];
    assign rl_pt = {t, 32'd0};
  end else begin : g_nrlt
    assign rl_pt = '0;
  end
  // the TMEM request arrays, each assembled in one process from the units' ports
  always_comb begin
    rq_en = '0; r_addr = '0; wq_en = '0; w_addr = '0; w_data = '0;
    rq_en[P_DMA] = dma_ren;   r_addr[P_DMA] = dma_raddr;
    rq_en[P_MXU][ULANES-1:0] = mxu_ren;   r_addr[P_MXU][ULANES-1:0] = mxu_raddr;
    rq_en[P_Q][ULANES-1:0] = q_ren;       r_addr[P_Q][ULANES-1:0] = q_raddr;
    rq_en[P_Q2][ULANES-1:0] = q_ren2;     r_addr[P_Q2][ULANES-1:0] = q_raddr2;
    rq_en[P_Q3][0] = q3_en;   r_addr[P_Q3][0] = q3_addr;
    rq_en[P_VA] = va_ren;     r_addr[P_VA] = va_raddr;
    rq_en[P_VB] = vb_ren;     r_addr[P_VB] = vb_raddr;
    rq_en[P_COLL] = coll_ren; r_addr[P_COLL] = coll_raddr;
    if (rl_pq) begin rq_en[P_COLL][0] = 1'b1; r_addr[P_COLL][0] = rl_a; end
    wq_en[W_DMA] = dma_wen;   w_addr[W_DMA] = dma_waddr;   w_data[W_DMA] = dma_wdata;
    wq_en[W_MXU][ULANES-1:0] = mxu_wen;
    w_addr[W_MXU][ULANES-1:0] = mxu_waddr;
    w_data[W_MXU][ULANES-1:0] = mxu_wdata;
    wq_en[W_VPU] = v_wen;     w_addr[W_VPU] = v_waddr;     w_data[W_VPU] = v_wdata;
    wq_en[W_COLL] = coll_wen; w_addr[W_COLL] = coll_waddr; w_data[W_COLL] = coll_wdata;
  end

  assign urdy  = {!coll_req && !rl_busy, r_vpu, r_q, r_mxu, r_dma};
  assign udone = {coll_ack || rl_v, d_vpu, d_q, d_mxu, d_dma};

  // ---- TMEM bank arbiter: all-or-nothing grants in priority order
  function automatic logic [NRP-1:0] grp_rports(input int g);
    case (g)
      G_DMA:  return NRP'(1) << P_DMA;
      G_COLL: return NRP'(1) << P_COLL;
      G_MXU:  return NRP'(1) << P_MXU;
      G_Q:    return (NRP'(1) << P_Q) | (NRP'(1) << P_Q2) | (NRP'(1) << P_Q3);
      default: return (NRP'(1) << P_VA) | (NRP'(1) << P_VB);
    endcase
  endfunction
  function automatic logic [NWP-1:0] grp_wports(input int g);
    case (g)
      G_DMA:  return NWP'(1) << W_DMA;
      G_COLL: return NWP'(1) << W_COLL;
      G_MXU:  return NWP'(1) << W_MXU;
      G_Q:    return '0;
      default: return NWP'(1) << W_VPU;
    endcase
  endfunction

  // The board build (TMEM replicated per read port, RPB >= NRP * LANES: reads never conflict;
  // WPB = 1; ARB_MASK) only needs write-bank masks: a unit is granted when no bank it writes was taken by
  // a higher-priority unit this cycle (a unit's own lanes write distinct banks). Shallow logic,
  // no counters. WPB 2 (the pumped one copy): when no bank it writes was taken by two of them
  // (acf2, the groups' triples). Other configurations count reads and writes per bank.
  logic [NG-1:0] gnt_m, gnt_c;
  logic          cgl_m, cgl_c;
  // Each group's write-bank mask, the pairwise conflicts between groups (in parallel), then
  // priority on single bits: a group is admitted unless it conflicts with an earlier admitted
  // group (the same result as accumulating the admitted groups' banks in priority order, with
  // the per-bank reductions out of the priority chain)
  logic [NG-1:0][LANES-1:0] amk;
  logic [NG-1:0][NG-1:0]    acf;
  logic [NG-1:0][NG-1:0][NG-1:0] acf2;   // WPB 2: [g][h][i] (i < h < g) the three write a bank
  logic [NG-1:0]            aok;
  always_comb begin
    logic [NWP-1:0] wp;
    for (int g = 0; g < NG; g++) begin
      wp = grp_wports(g);
      amk[g] = '0;
      for (int p = 0; p < NWP; p++)
        if (wp[p])
          for (int l = 0; l < LANES; l++)
            if (wq_en[p][l]) amk[g][w_addr[p][l][BW-1:0]] = 1'b1;
    end
    acf = '0;
    acf2 = '0;
    for (int g = 0; g < NG; g++)
      for (int h = 0; h < g; h++) begin
        acf[g][h] = (amk[g] & amk[h]) != '0;
        for (int i = 0; i < h; i++) acf2[g][h][i] = (amk[g] & amk[h] & amk[i]) != '0;
      end
  end
  // the guests on the DMA's TMEM copy read only when no port before them there asks (from the
  // requests: the copy's block RAM address does not wait for the grants)
  // (TMEM_IMPL 1: no guests, the banked arbiter counts every copy's banks)
  localparam bit GUESTS = TMEM_IMPL == 0 || (TMEM_IMPL == 3 && TMEM_GST);
  wire q3_blk = GUESTS && (|rq_en[P_Q3]) && (|rq_en[P_DMA]);
  wire coll_blk = GUESTS && (|rq_en[P_COLL]) && ((|rq_en[P_DMA]) || (|rq_en[P_Q3]));
  // TMEM_RPL: a group whose port replays a lost read is held (tm_frz, a register); the
  // collective, which is not replayed, reads only where its request does not lose (tm_bx)
  logic [NG-1:0] rpl_hold, rpl_frz;
  always_comb
    for (int g = 0; g < NG; g++) begin
      rpl_frz[g] = TMEM_RPL && g != G_VPU && g != G_COLL && (tm_frz & grp_rports(g)) != '0;
      rpl_hold[g] = rpl_frz[g] || (TMEM_RPL && g == G_COLL && tm_bx[P_COLL] && |rq_en[P_COLL]);
    end
  always_comb begin
    for (int g = 0; g < NG; g++) begin
      aok[g] = !rpl_hold[g];
      if (WPB == 1) begin
        for (int h = 0; h < g; h++) if (aok[h] && acf[g][h]) aok[g] = 1'b0;
      end else
        for (int h = 0; h < g; h++)
          for (int i = 0; i < h; i++) if (aok[h] && aok[i] && acf2[g][h][i]) aok[g] = 1'b0;
      if (g == G_Q && q_blk) aok[g] = 1'b0;   // QST write the DRAM cannot take (q_blk)
      if ((g == G_Q && q3_blk) || (g == G_COLL && coll_blk)) aok[g] = 1'b0;
    end
    gnt_m = aok;
    gnt_m[G_COLL] = coll_gnt;
    cgl_m = aok[G_COLL];
  end

  logic [NG-1:0] gnt_k, gnt_q, gnt_x, gnt_s;
  logic          cgl_k, cgl_q, cgl_x, cgl_s;
  always_comb begin
    gnt = ARB_MASK ? gnt_m : (TMEM_IMPL == 1) ? (TMEM_RQ ? gnt_q : gnt_k) :
          (TMEM_IMPL == 2) ? gnt_x : (TMEM_IMPL == 3) ? gnt_s : gnt_c;
    coll_gnt_local = ARB_MASK ? cgl_m : (TMEM_IMPL == 1) ? (TMEM_RQ ? cgl_q : cgl_k) :
                     (TMEM_IMPL == 2) ? cgl_x : (TMEM_IMPL == 3) ? cgl_s : cgl_c;
    for (int p = 0; p < NRP; p++) r_en[p] = rq_en[p];
    for (int p = 0; p < NWP; p++) w_en[p] = wq_en[p];
    w_gnt = '1;
    w_gnt[W_MXU] = gnt[G_MXU];
    w_gnt[W_VPU] = gnt[G_VPU];
    w_gnt[W_COLL] = gnt[G_COLL];
    if (!gnt[G_MXU])  begin if (!MXU_TQ) r_en[P_MXU] = '0; w_en[W_MXU] = '0; end
    if (MXU_TQ && rpl_frz[G_MXU]) r_en[P_MXU] = '0;   // the MXU's read waits out its replay
    if (!gnt[G_Q])    begin r_en[P_Q] = '0; r_en[P_Q2] = '0; r_en[P_Q3] = '0; end
    if (!v_ren)       begin r_en[P_VA] = '0; r_en[P_VB] = '0; end
    if (!gnt[G_VPU])  w_en[W_VPU] = '0;
    if (!gnt[G_COLL]) begin r_en[P_COLL] = '0; w_en[W_COLL] = '0; end
    if (TMEM_IMPL == 3 && !TMEM_RPL)  // a split group's ports: read once, ahead of the grant
      for (int p = 0; p < NRP; p++) begin
        if (tm_sv[p]) r_en[p] = '0;
        if (tm_pre[p]) r_en[p] = rq_en[p];
      end
  end

  // TMEM_IMPL 3 (EXPERIMENTAL, docs/tmem_nocopy.md): one copy of sub-banked RAMs (otpu_tmem IMPL
  // 3). A group is granted when no port of another group blocks one of its reads (tm_bx, from
  // the requests: the RAMs' addresses do not wait for the grants), its write banks are free (acf,
  // as gnt_q) and its own ports do not collide (tm_bs). When they do, its ports read over
  // successive cycles: those not blocked read ahead of the grant (tm_pre: their lanes hold the
  // words), drop out of the requests (tm_sv), and the unit is granted when the rest can read.
  // (TMEM_RPL: the requests of a group held for a replay, and the VPU's while its registered
  // enable is low, ask no RAM.) TMEM_RPL with guests: the collective asks only when the guests'
  // rule lets it read (coll_blk, from the requests). While QUANT holds for a replay its QUANT 3
  // request asks nothing but is still up, and coll_blk keeps the collective from reading; had the
  // collective asked, the DMA's asker, first in rank, would ask for its words every cycle, and a
  // replay on that RAM side (the one holding QUANT among them) would lose to it for good. Every
  // ask of the DMA's asker is then a read that happens, unless the read waits on something no
  // replay holds up (the DRAM's write room for a QST, another slice's grant of the collective,
  // the DMA's write banks), and a replay waits only on it and on older replays (otpu_tmem)
  always_comb begin
    logic [NRP-1:0] off;
    off = '0;
    if (TMEM_IMPL == 3 && TMEM_RPL)
      for (int g = 0; g < NG; g++)
        if ((g == G_VPU) ? !v_ren : rpl_frz[g]) off = off | grp_rports(g);
    if (TMEM_IMPL == 3 && TMEM_RPL && coll_blk) off[P_COLL] = 1'b1;
    for (int p = 0; p < NRP; p++)
      rq_t[p] = ((TMEM_IMPL == 3 && !TMEM_RPL && tm_sv[p]) || off[p]) ? '0 : rq_en[p];
    for (int p = 0; p < NRP; p++)
      r_ben[p] = ((TMEM_IMPL == 3 && !TMEM_RPL && tm_sv[p]) || off[p]) ? '0 : rq_b[p];
  end
  logic [NG-1:0] gs3;
  always_comb begin
    gnt_s = gs3;
    gnt_s[G_COLL] = coll_gnt;      // every slice must grant the collective
  end
  always_comb begin
    logic [NG-1:0]  gs;
    logic [NRP-1:0] rp, ask;
    logic           rd_ok, self;
    tm_pre = '0;
    for (int p = 0; p < NRP; p++) ask[p] = |rq_t[p];
    for (int g = 0; g < NG; g++) begin
      rp = grp_rports(g);
      rd_ok = (tm_bx & rp & ask) == '0;
      self = (tm_bs & rp & ask) != '0;
      gs[g] = rd_ok && !self;
      for (int h = 0; h < g; h++) if (gs[h] && acf[g][h]) gs[g] = 1'b0;
      if (g == G_Q && q_blk) gs[g] = 1'b0;    // QST write the DRAM cannot take (q_blk)
      if (rd_ok && self) tm_pre = tm_pre | (rp & ask & ~tm_bs);
    end
    gs3 = gs;
    cgl_s = gs[G_COLL];
  end
  always_ff @(posedge clk)
    if (srst || TMEM_IMPL != 3 || TMEM_RPL) tm_sv <= '0;
    else
      for (int g = 0; g < NG; g++) begin
        logic [NRP-1:0] rp;
        rp = grp_rports(g);
        for (int p = 0; p < NRP; p++)
          if (rp[p]) tm_sv[p] <= gnt[g] ? 1'b0 : (tm_sv[p] | tm_pre[p]);
      end

  always_comb begin
    int rc [LANES];
    int wc [LANES];
    int ur [LANES];
    int uw [LANES];
    logic ok;
    logic [NRP-1:0] rp;
    logic [NWP-1:0] wp;
    for (int b = 0; b < LANES; b++) begin rc[b] = 0; wc[b] = 0; end
    cgl_c = 1'b1;
    for (int g = 0; g < NG; g++) begin
      rp = grp_rports(g);
      wp = grp_wports(g);
      for (int b = 0; b < LANES; b++) begin ur[b] = 0; uw[b] = 0; end
      for (int p = 0; p < NRP; p++)
        if (rp[p])
          for (int l = 0; l < LANES; l++)
            if (rq_en[p][l]) ur[r_addr[p][l][BW-1:0]] += 1;
      for (int p = 0; p < NWP; p++)
        if (wp[p])
          for (int l = 0; l < LANES; l++)
            if (wq_en[p][l]) uw[w_addr[p][l][BW-1:0]] += 1;
      ok = 1'b1;
      for (int b = 0; b < LANES; b++)
        if (rc[b] + ur[b] > RPB || wc[b] + uw[b] > WPB) ok = 1'b0;
      if (g == G_Q && q_blk) ok = 1'b0;    // QST write the DRAM cannot take (q_blk)
      if ((g == G_Q && q3_blk) || (g == G_COLL && coll_blk)) ok = 1'b0;
      if (g == G_COLL) begin
        cgl_c = ok;
        gnt_c[g] = coll_gnt;      // every slice must grant the collective
      end else begin
        gnt_c[g] = ok;
      end
      if (ok)
        for (int b = 0; b < LANES; b++) begin rc[b] += ur[b]; wc[b] += uw[b]; end
    end
  end

  // TMEM_IMPL 2 (simulation): a candidate's read arbitration (otpu_tmem_model, plusargs)
`ifndef SYNTHESIS
  if (TMEM_IMPL == 2) begin : g_model
    otpu_tmem_model #(.LANES(LANES), .NRP(NRP), .NWP(NWP), .NG(NG),
                      //      COLL    Q2      Q3      VB      VA      Q       DMA     MXU
                      .RGRP({3'(G_COLL), 3'(G_Q), 3'(G_Q), 3'(G_VPU), 3'(G_VPU), 3'(G_Q),
                             3'(G_DMA), 3'(G_MXU)}),
                      //      VPU       COLL       MXU       DMA
                      .WGRP({3'(G_VPU), 3'(G_COLL), 3'(G_MXU), 3'(G_DMA)}),
                      .GCOLL(G_COLL), .SID(SID)) u_model (
      .clk, .rst(srst), .run(!srst && !halted), .rq_en, .r_addr, .wq_en, .w_addr,
      .r_take(r_en), .w_take(w_en), .blk(NG'(q_blk) << G_Q), .coll_gnt,
      .gnt(gnt_x), .cgl(cgl_x));
  end else begin : g_nomodel
    assign gnt_x = '0;
    assign cgl_x = 1'b0;
  end
`else
  assign gnt_x = '0;
  assign cgl_x = 1'b0;
`endif

  // The banked TMEM's arbiter (TMEM_IMPL 1): as gnt_c, but each copy's bank takes one read per
  // cycle, and a port's lanes in one bank count once (they read one word there)
  always_comb begin
    logic [NCP-1:0][LANES-1:0] rk, uk;
    logic [LANES-1:0]          wk, vk;
    logic ok;
    logic [NRP-1:0] rp;
    logic [NWP-1:0] wp;
    rk = '0; wk = '0;
    cgl_k = 1'b1;
    gnt_k = '0;
    for (int g = 0; g < NG; g++) begin
      rp = grp_rports(g);
      wp = grp_wports(g);
      uk = '0; vk = '0;
      ok = 1'b1;
      for (int p = 0; p < NRP; p++)
        if (rp[p])
          for (int l = 0; l < LANES; l++)
            if (rq_en[p][l]) uk[CPY[p]][r_addr[p][l][BW-1:0]] = 1'b1;
      for (int p = 0; p < NWP; p++)
        if (wp[p])
          for (int l = 0; l < LANES; l++)
            if (wq_en[p][l]) vk[w_addr[p][l][BW-1:0]] = 1'b1;
      if ((rk & uk) != '0 || (wk & vk) != '0) ok = 1'b0;
      if (g == G_Q && q_blk) ok = 1'b0;    // QST write the DRAM cannot take (q_blk)
      if (g == G_COLL) begin
        cgl_k = ok;
        gnt_k[g] = coll_gnt;      // every slice must grant the collective
      end else begin
        gnt_k[g] = ok;
      end
      if (ok) begin rk |= uk; wk |= vk; end
    end
  end

  // The banked TMEM with request-priority addresses (TMEM_IMPL 1, TMEM_RQ): a group's reads are
  // blocked where a group before it asks for the same copy and bank, granted or not (each copy's
  // bank address is the first asking port's: otpu_tmem RQ), so the RAMs' addresses come from the
  // requests alone and the grants reach no RAM pin. Its write banks are admitted as gnt_m's, a
  // chain of single bits over the pairwise conflicts (acf).
  logic [NG-1:0][NCP-1:0][LANES-1:0] qk;
  logic [NG-1:0]                     qblk;
  always_comb begin
    logic [NRP-1:0] rp;
    for (int g = 0; g < NG; g++) begin
      rp = grp_rports(g);
      qk[g] = '0;
      for (int p = 0; p < NRP; p++)
        if (rp[p])
          for (int l = 0; l < LANES; l++)
            if (rq_en[p][l]) qk[g][CPY[p]][r_addr[p][l][BW-1:0]] = 1'b1;
    end
    for (int g = 0; g < NG; g++) begin
      qblk[g] = 1'b0;
      for (int h = 0; h < g; h++) if ((qk[g] & qk[h]) != '0) qblk[g] = 1'b1;
    end
  end
  always_comb begin
    logic [NG-1:0] ok;
    for (int g = 0; g < NG; g++) begin
      ok[g] = !qblk[g];
      for (int h = 0; h < g; h++) if (ok[h] && acf[g][h]) ok[g] = 1'b0;
      if (g == G_Q && q_blk) ok[g] = 1'b0;    // QST write the DRAM cannot take (q_blk)
    end
    gnt_q = ok;
    gnt_q[G_COLL] = coll_gnt;      // every slice must grant the collective
    cgl_q = ok[G_COLL];
  end

  // ---- the SW port's output register (docs/asic.md "The ports"): the QST writes leave the slice
  // from flip-flops, not through the quantizer's scale-row words and the TMEM grant (on the card
  // the quantizer -> the DRAM adapter's gather registers: 5-14 levels, 4-7 ns of route; -0.260 ns
  // in an r2 candidate, -0.092 ns in an r2.1 one). The register takes a write when it is empty or
  // drains this cycle (sw_rdy_i, what the arbiters see: sw_rdy does not depend on sw_req), and a
  // write in it counts as one the DRAM has not taken (wr_idle_i), so a QST, a HALT CHAIN and a
  // WAITW timeout still end only once the DRAM has every write (and a write in port B's output
  // register, below, counts the same). sw_req is its valid bit: outside the slice, every write it
  // gave is taken once wr_idle && !sw_req && !(b_req && b_we) (otpu_board's STATUS WR_IDLE and
  // go_ok). Only the loader's reset clears it: a write the quantizer gave before a run's reset
  // (rst) still goes out, as one in the DRAM's queues does. The units and the arbiters read the
  // DRAM's sw_rdy and wr_idle only through it (tests/test_target.py)
  wire  sw_req_c = q_areq && q_awe;
  logic sw_v;
  assign sw_rdy_i  = !sw_v || sw_rdy;
  assign wr_idle_i = wr_idle && !sw_v && !(b_req && b_we);
  assign sw_req    = sw_v;
  always_ff @(posedge clk) begin : sw_reg
    if (sys_rst) sw_v <= 1'b0;
    else if (sw_rdy_i) sw_v <= sw_req_c;
    if (sw_rdy_i && sw_req_c) begin
      sw_addr <= q_aaddr; sw_wdata <= q_awdata; sw_be <= q_abe;
    end
  end : sw_reg
`ifndef SYNTHESIS
  // the arbiters withhold the quantizer's grant while the port cannot take its write
  always @(posedge clk)
    if (!sys_rst && sw_req_c && !sw_rdy_i)
      $fatal(1, "otpu_slice: a QST write the SW port cannot take");
`endif

  // ---- DRAM ports: the MXU's port A reads, and port B's requests (b_*_c), into their output
  // registers (below)
  logic           b_req_c, b_tag_c, b_we_c;
  logic [CL-1:0]  b_par_c;
  logic [D/4-1:0] b_wmask_c;
  logic [D*8-1:0] b_wdata_c;
  logic [WA-1:0]  b_addr_c;
  assign mxu_bgnt = !dma_breq && b_rdy_i && !ic_req;
  assign mxu_agnt = a_rdy_i;
  assign a_we    = 1'b0;               // port A carries the MXU's scale reads only
  assign a_wdata = '0;
  assign a_be    = '0;
  always_comb begin
    b_req_c   = dma_breq | mxu_breq;
    b_tag_c   = dma_breq;
    b_we_c    = dma_bwe;
    b_wmask_c = dma_bwmask;
    b_wdata_c = dma_bwdata;
    b_addr_c  = dma_breq ? dma_baddr : mxu_baddr;
    // the address's parity from each source's own address (their registers), muxed with it:
    // the parity of the muxed address ran on from the DMA's request and the MXU's grant into
    // every write of otpu_native_dram's queues (DMA occ -> qbm, 11 levels, the 959b425 build's
    // worst DMA -> memory paths, -0.390 ns at 133.33 MHz); it now ends at the output register
    b_par_c   = dma_breq ? chash(dma_baddr) : chash(mxu_baddr);
    if (ld_busy && !ICACHE) begin     // the units are held in reset
      b_req_c = ld_req; b_tag_c = 1'b1; b_we_c = 1'b0;
      b_addr_c = WA'(ld_a); b_par_c = chash(WA'(ld_a));      // (a program below 4 GiB)
    end
    if (ic_req) begin                 // ICACHE: a line fill's read, ahead of the units'
      b_req_c = 1'b1; b_tag_c = 1'b1; b_we_c = 1'b0;
      b_addr_c = WA'(ic_addr); b_par_c = chash(WA'(ic_addr));
    end
  end
  // port W: the MXU's grouped operands (weight groups, WG > 1; otpu_mxu)
  assign pw_req  = WG > 1 && mxu_pwreq;
  assign pw_addr = mxu_pwaddr;

  // ---- port B's registers (docs/asic.md "The ports"): the requests (b_req ... b_par) leave the
  // slice from flip-flops and the read data (b_rvalid, b_rtag, b_rdata) enters through
  // flip-flops, so neither the DMA's and the MXU's request logic nor the units' read paths reach
  // the boundary (the ASIC card point: the DMA's write-data rotator -> b_wdata, -0.59 ns at
  // 2.5 ns with the clock's insertion delay credited, -2.50 ns against the ideal I/O clock; the
  // card's FPGA: the DMA -> the adapter's queue RAM, +0.162 ns on r2.1's E3, 6.4 ns of route).
  // The output register takes a request when it is empty or drains this cycle (b_rdy_i, what
  // the DMA, the MXU and the loader see: b_rdy does not depend on b_req), b_req its valid bit.
  // A write in it is one the DRAM has not taken (wr_idle_i, with the SW register's), a read in it
  // or its data in the input register one not back (rd_idle_i, with port A's: the loader's
  // start). Only the
  // loader's reset clears them: a request given before a run's reset (rst) still goes out, and
  // its data comes back, as one in the DRAM's queues. The units, the loader and the arbiters read
  // the DRAM's b_rdy, rd_idle, b_rvalid, b_rtag and b_rdata only through them
  // (tests/test_target.py). One cycle more each way costs nothing measurable (co-simulated:
  // within +-0.07% at m4 on decode, a tiled prompt run and a MoE step, signs mixed)
  assign b_rdy_i   = !b_req || b_rdy;
  assign rd_idle_i = rd_idle && !a_req && !(b_req && !b_we) && !b_rvalid_i;

  // ---- ICACHE: the fills' chunks among port B's reads. The memory returns port B's reads in
  // order, whatever their tag: rd_n counts the reads into port B's output register, ifq keeps the
  // place of each fill's first read (a fill's IC_NB reads go in one after the other: ahead of the
  // units'), rt_n counts the reads back, so the next one back is a fill's chunk (ic_in, from
  // flip-flops) while it is one of the IC_NB from the oldest fill's first. Its data and ic_rv_i
  // come through the input register together (it is part of it: the memory's raw returns, as
  // pb_reg's). ic_inv starts them over: no read in flight then
  localparam int IC_NB = 1024 / D + (SEQ_WC ? 256 / D : 0);
  if (ICACHE) begin : g_icrd
    logic [15:0] rd_n, rt_n;
    logic [15:0] ifq [4];
    logic [1:0]  ifq_h, ifq_t;
    logic [5:0]  ic_k;                // the fill's reads taken
    logic [2:0]  ifq_c;               // fills with a read in and a chunk to come
    wire  [15:0] rk = rt_n - ifq[ifq_h];
    wire         ic_in = ifq_c != '0 && rk < 16'(IC_NB);
    wire         push = ic_req && b_rdy_i && ic_k == '0;
    wire         pop = b_rvalid && ic_in && 32'(rk) == IC_NB - 1;
    always_ff @(posedge clk) begin
      ic_rv_i <= !sys_rst && b_rvalid && ic_in;
      if (ic_inv) begin
        rd_n <= '0; rt_n <= '0; ifq_h <= '0; ifq_t <= '0; ifq_c <= '0; ic_k <= '0;
      end else begin
        if (b_rdy_i && b_req_c && !b_we_c) rd_n <= rd_n + 1'b1;
        if (ic_req && b_rdy_i) ic_k <= (32'(ic_k) == IC_NB - 1) ? '0 : ic_k + 1'b1;
        if (push) begin ifq[ifq_t] <= rd_n; ifq_t <= ifq_t + 1'b1; end
        if (b_rvalid) rt_n <= rt_n + 1'b1;
        if (pop) ifq_h <= ifq_h + 1'b1;
        ifq_c <= ifq_c + 3'(push) - 3'(pop);
      end
    end
  end else begin : g_noicrd
    assign ic_rv_i = 1'b0;
  end

  always_ff @(posedge clk) begin : pb_reg
    if (sys_rst) b_req <= 1'b0;
    else if (b_rdy_i) b_req <= b_req_c;
    if (b_rdy_i && b_req_c) begin
      b_tag <= b_tag_c; b_we <= b_we_c; b_wmask <= b_wmask_c; b_wdata <= b_wdata_c;
      b_addr <= b_addr_c; b_par <= b_par_c;
    end
    b_rvalid_i <= !sys_rst && b_rvalid;
    // every cycle, not on b_rvalid: every reader takes them in a b_rvalid_i cycle only (the
    // loader's IMEM and sidecar writes, the DMA's line buffer, the MXU's FIFO), so they are the
    // same there, and b_rvalid (the DRAM's logic) enables no 1,025 flip-flops
    b_rtag_i <= b_rtag; b_rdata_i <= b_rdata;
  end : pb_reg

  // ---- port A's register (docs/asic.md "The ports"): the MXU's scale reads (a_req, a_addr) leave
  // the slice from flip-flops, so the MXU's issue (the DMA's request -> port B's grant -> the
  // MXU's issue -> a_req: the ASIC card point's a_req, -1.62 ns against the ideal I/O clock), and
  // a_rdy and b_rdy through it, do not reach the boundary. It is the register the card's DRAM
  // adapter took port A's requests into (otpu_native_dram), moved here: one register between the
  // MXU and the adapter's queues on the card as before, so the same cycles. It takes a request
  // when it is empty or drains this cycle (a_rdy_i, the MXU's grant: a_rdy does not depend on
  // a_req), a_req its valid bit; a read in it is one not back (rd_idle_i). Only the loader's
  // reset clears it, as the adapter's did: it holds a request, not data, and the adapter's
  // a_flush (a run's reset, a load, a WAITW's word holding) drops the beats it holds (its
  // reused beat and runs), so a read that leaves after a flush reads the memory as the host left
  // it. The read data needs none: the MXU registers a_rdata at the port (ln1)
  assign a_rdy_i = !a_req || a_rdy;
  always_ff @(posedge clk) begin : pa_reg
    if (sys_rst) a_req <= 1'b0;
    else if (a_rdy_i) a_req <= mxu_areq;
    if (a_rdy_i && mxu_areq) a_addr <= mxu_aaddr;
  end : pa_reg

  // a unit's DRAM request waiting for its port, a cycle late (pf: the board's B_STALL and
  // DRAM_WAIT): port B's (the DMA, the MXU stream, the loader, which runs in reset), port A's (the
  // MXU's scales), the quantizer's QST write. A request parked in the SW, port A or port B register
  // while the DRAM refuses it is not one: the unit that gave it has moved on (an MXU at its stream's
  // tag limit waits for its data, not for the port)
  logic bw_r, dw_r;
  always_ff @(posedge clk) begin
    bw_r <= !sys_rst && b_req_c && !b_rdy_i;
    dw_r <= !sys_rst && ((b_req_c && !b_rdy_i) || (mxu_areq && !a_rdy_i) || (q_awant && !sw_rdy_i));
  end

  // ---- activity and trace events (pf; docs/observability.md). The sequencer's events come
  // registered (seq_ev_t), the units report their counters the cycle after an instruction ends,
  // and the port and stall counters of the H, P and Q trace lines live here:
  //   P  per window of `win` cycles: DRAM port B requests (MXU bm, DMA bd), port A requests
  //      (MXU am, QST aq), MXU compute (mx), and cycles the MXU drain, QUANT, VPU and collective
  //      lost to TMEM bank arbitration (fm fq fv fc)
  //   Q  the same window: a unit's port B / port A or QST request waited for its port (bs, as;
  //      as dwait above: not one parked in a port's register), the MXU had work
  //      but no chunk (ms) or chunks but did not consume (mb), the summed MXU chunk-FIFO level
  //      (ff; average = ff / n), the program loader used port B (ld)
  //   H  at the halt: the port requests since reset (bmxu bdma amxu aq)
  // Every cycle's activity is registered first (a*_r) and the sums run a cycle behind, so no
  // counter adds to an arbitration or DRAM-ready path: a window's sums are s + a_r the cycle
  // after its last cycle (pf.w), the halt totals the cycle after the halt (pf.h). Counting
  // stops the cycle after the halt. In simulation, +trace prints the trace lines from pf.
  // a unit "loses" a cycle when it requested TMEM ports and was not granted
  wire lose_mxu = !gnt[G_MXU] && ((!MXU_TQ && (|rq_en[P_MXU])) || (|wq_en[W_MXU]));
  wire lose_q   = !gnt[G_Q] && ((|rq_en[P_Q]) || (|rq_en[P_Q2]) || q3_en);
  wire lose_vpu = !gnt[G_VPU] && ((|rq_en[P_VA]) || (|rq_en[P_VB]) || (|wq_en[W_VPU]));
  wire lose_col = !gnt[G_COLL] && ((|rq_en[P_COLL]) || (|wq_en[W_COLL]));

`ifndef SYNTHESIS
  // TMEM read demand (docs/tmem_pump.md; simulation only, a TMEMSTAT line at the end): each cycle
  // out of reset and before the halt, how many read streams ask (MXU; the DMA's copy: DMA,
  // QUANT 3, COLL; QUANT; VPU A; VPU B; QUANT 2: 0..6), the most read ports asking for one bank
  // (0..8), the same with a write asking for that bank (0..9), the cycles the MXU, QUANT, the
  // VPU and the collective lost to the arbitration, and each read port's reads (rd: cycles it
  // reads) and lost reads (rpl: TMEM_RPL's replays, tm_lost)
  longint ts_cyc, ts_lose [4], ts_n [7], ts_d [NRP + 1], ts_dw [NRP + 2], ts_rd [NRP], ts_rpl [NRP];
  initial begin
    ts_cyc = 0;
    for (int i = 0; i < NRP; i++) begin ts_rd[i] = 0; ts_rpl[i] = 0; end
    for (int i = 0; i < 4; i++) ts_lose[i] = 0;
    for (int i = 0; i < 7; i++) ts_n[i] = 0;
    for (int i = 0; i <= NRP; i++) ts_d[i] = 0;
    for (int i = 0; i <= NRP + 1; i++) ts_dw[i] = 0;
  end
  always @(posedge clk)
    if (!srst && !halted) begin
      int n, dm, dwm, c;
      logic [LANES-1:0] wb;
      logic hit;
      n = int'(|rq_en[P_MXU]) + int'((|rq_en[P_DMA]) || (|rq_en[P_Q3]) || (|rq_en[P_COLL])) +
          int'(|rq_en[P_Q]) + int'(|rq_en[P_VA]) + int'(|rq_en[P_VB]) + int'(|rq_en[P_Q2]);
      wb = '0;
      for (int p = 0; p < NWP; p++)
        for (int l = 0; l < LANES; l++) if (wq_en[p][l]) wb[w_addr[p][l][BW-1:0]] = 1'b1;
      dm = 0; dwm = 0;
      for (int b = 0; b < LANES; b++) begin
        c = 0;
        for (int p = 0; p < NRP; p++) begin
          hit = 1'b0;
          for (int l = 0; l < LANES; l++)
            if (rq_en[p][l] && r_addr[p][l][BW-1:0] == BW'(b)) hit = 1'b1;
          c += int'(hit);
        end
        if (c > dm) dm = c;
        if (c + int'(wb[b]) > dwm) dwm = c + int'(wb[b]);
      end
      ts_cyc++; ts_n[n]++; ts_d[dm]++; ts_dw[dwm]++;
      for (int p = 0; p < NRP; p++) begin
        if (|r_en[p]) ts_rd[p]++;
        if (tm_lost[p]) ts_rpl[p]++;
      end
      if (lose_mxu) ts_lose[0]++;
      if (lose_q)   ts_lose[1]++;
      if (lose_vpu) ts_lose[2]++;
      if (lose_col) ts_lose[3]++;
    end
  final begin
    string s;
    s = $sformatf("TMEMSTAT sid=%0d cyc=%0d lose=%0d,%0d,%0d,%0d n=", SID, ts_cyc, ts_lose[0],
                  ts_lose[1], ts_lose[2], ts_lose[3]);
    for (int i = 0; i < 7; i++) s = {s, $sformatf("%s%0d", (i > 0) ? "," : "", ts_n[i])};
    s = {s, " d="};
    for (int i = 0; i <= NRP; i++) s = {s, $sformatf("%s%0d", (i > 0) ? "," : "", ts_d[i])};
    s = {s, " dw="};
    for (int i = 0; i <= NRP + 1; i++) s = {s, $sformatf("%s%0d", (i > 0) ? "," : "", ts_dw[i])};
    s = {s, " rd="};
    for (int i = 0; i < NRP; i++) s = {s, $sformatf("%s%0d", (i > 0) ? "," : "", ts_rd[i])};
    s = {s, " rpl="};
    for (int i = 0; i < NRP; i++) s = {s, $sformatf("%s%0d", (i > 0) ? "," : "", ts_rpl[i])};
    $display("%s", s);
  end

  // The TMEM's accesses, cycle by cycle (docs/tmem_nocopy.md; +tmtrace=<file>, appended; S > 1:
  // <file>.<sid>): records of three little-endian 32-bit words {cycle, info, value}, from reset
  // to the halt. info[3:0] the kind, [7:4] the port or unit, [15:8] the lanes:
  //   0 a read that happened (r_en), value lane 0's address (lane l reads value + l)
  //   1 a write that happened (w_en), the same
  //   2 / 3 a read / write lane of the MXU's ports (scattered lanes): one record per lane,
  //     value its address
  //   4 an instruction starts on unit [7:4]: [15:8] flags, [23:16] VOP func (w6[23:16]),
  //     [31:24] opcode; value the sequencer's instruction count
  //   5 an instruction completes on unit [7:4]
  //   6 a unit asked for TMEM and was not granted: [15:8] a mask by group (DMA, COLL, VPU, MXU,
  //     QUANT)
  //   15 a simulation starts (cycle 0)
  int     tt_fd;
  longint tt_cyc;
  initial begin
    string fn;
    tt_fd = 0;
    tt_cyc = 0;
    if ($value$plusargs("tmtrace=%s", fn)) begin
      if (S > 1) fn = $sformatf("%s.%0d", fn, SID);
      tt_fd = $fopen(fn, "ab");
      if (tt_fd == 0) $fatal(1, "otpu_slice: cannot open %s", fn);
      $fwrite(tt_fd, "%u%u%u", 32'd0, 32'd15, 32'd0);
    end
  end
  /* verilator lint_off BLKSEQ */
  always @(posedge clk) begin
    tt_cyc <= tt_cyc + 1;
    if (tt_fd != 0 && !srst && !halted) begin
      logic [31:0] cy;
      logic [NG-1:0] lg;
      cy = 32'(tt_cyc);
      for (int p = 0; p < NRP; p++)
        if (|r_en[p]) begin
          if (p == P_MXU) begin
            for (int l = 0; l < LANES; l++)
              if (r_en[p][l]) $fwrite(tt_fd, "%u%u%u", cy, {16'd0, 8'(1 << l), 4'(p), 4'd2}, r_addr[p][l]);
          end else
            $fwrite(tt_fd, "%u%u%u", cy, {16'd0, 8'(r_en[p]), 4'(p), 4'd0}, r_addr[p][0]);
        end
      for (int p = 0; p < NWP; p++)
        if (|w_en[p]) begin
          if (p == W_MXU) begin
            for (int l = 0; l < LANES; l++)
              if (w_en[p][l]) $fwrite(tt_fd, "%u%u%u", cy, {16'd0, 8'(1 << l), 4'(p), 4'd3}, w_addr[p][l]);
          end else
            $fwrite(tt_fd, "%u%u%u", cy, {16'd0, 8'(w_en[p]), 4'(p), 4'd1}, w_addr[p][0]);
        end
      for (int u = 0; u < NUNITS; u++) begin
        if (ustart[u])
          $fwrite(tt_fd, "%u%u%u", cy, {ucmd[u].op, ucmd[u].w6[23:16], ucmd[u].flags, 4'(u), 4'd4},
                  sq_icount);
        if (udone[u]) $fwrite(tt_fd, "%u%u%u", cy, {24'd0, 4'(u), 4'd5}, 32'd0);
      end
      lg = '0;
      if (lose_mxu) lg[G_MXU] = 1'b1;
      if (lose_q)   lg[G_Q] = 1'b1;
      if (lose_vpu) lg[G_VPU] = 1'b1;
      if (lose_col) lg[G_COLL] = 1'b1;
      if (lg != '0) $fwrite(tt_fd, "%u%u%u", cy, {16'd0, 8'(lg), 4'd0, 4'd6}, 32'd0);
    end
  end
  /* verilator lint_on BLKSEQ */
  final if (tt_fd != 0) $fclose(tt_fd);
`endif

  logic [31:0] win;                  // cycles per window
`ifdef SYNTHESIS
  assign win = 32'(PQ_WIN);
`else
  int bucket;
  initial begin
    if (!$value$plusargs("bucket=%d", bucket)) bucket = PQ_WIN;
    if (bucket < 1 || bucket >= (1 << 24)) $fatal(1, "otpu_slice: bucket must be 1 .. 2^24 - 1");
  end
  assign win = 32'(bucket);
`endif
  initial if (PQ_WIN < 1 || PQ_WIN >= (1 << 24)) $fatal(1, "otpu_slice: PQ_WIN out of range");

  // this cycle's activity: the P fields, the Q fields but ff (bs as ms mb ld), the FIFO level
  logic [NP-1:0] ap, ap_r;
  logic [4:0]    aq, aq_r;
  logic [FW-1:0] af_r;
  assign ap = {lose_col, lose_vpu, lose_q, lose_mxu, mxu_pop, q_areq, mxu_areq, dma_breq,
               mxu_breq};
  assign aq = {ld_busy && ld_req, mxu_block, mxu_starve,
               (mxu_areq && !a_rdy_i) || (q_awant && !sw_rdy_i), b_req_c && !b_rdy_i};
  logic [31:0] w_n, n_r;             // cycles of the current window before this one; last n
  logic        h_d, we_r, h_ev;      // halted a cycle ago; a window ended / the halt, last cycle
  logic [NP-1:0][31:0] sp;           // window sums, a cycle behind
  logic [NQ-1:0][31:0] sq;
  logic [NH-1:0][31:0] sh;           // totals since reset, a cycle behind
  wire w_end = !h_d && (w_n + 1 == win || halted);
  always_ff @(posedge clk) begin
    if (rst) begin
      h_d <= 1'b0; we_r <= 1'b0; h_ev <= 1'b0; w_n <= '0;
      ap_r <= '0; aq_r <= '0; af_r <= '0;
      sp <= '0; sq <= '0; sh <= '0;
    end else begin
      h_d <= halted;
      h_ev <= halted && !h_d;
      we_r <= w_end;
      n_r <= w_n + 1;
      if (!h_d) w_n <= w_end ? '0 : w_n + 1;
      ap_r <= h_d ? '0 : ap;
      aq_r <= h_d ? '0 : aq;
      af_r <= h_d ? '0 : mxu_level;
      for (int k = 0; k < NP; k++) sp[k] <= we_r ? '0 : sp[k] + 32'(ap_r[k]);
      for (int k = 0; k < NQ; k++)
        sq[k] <= we_r ? '0 : sq[k] + ((k == 4) ? 32'(af_r) : 32'(aq_r[k < 4 ? k : 4]));
      for (int k = 0; k < NH; k++) sh[k] <= sh[k] + 32'(ap_r[k]);
    end
  end

  always_comb begin
    pf.sq = sq_ev;
    pf.mac = ap_r[4];
    pf.starve = aq_r[2];
    pf.deny = |ap_r[8:5];
    pf.bwait = bw_r;
    pf.dwait = dw_r;
    pf.u_mxu = mxu_u;
    pf.u_mxu_v = mxu_uv;
    pf.u_q = q_u;
    pf.u_q_frz = q_frz;
    pf.u_vpu = v_u;
    pf.u_vpu_frz = v_frz;
    pf.w = we_r;
    pf.w_n = n_r;
    for (int k = 0; k < NP; k++) pf.w_p[k] = sp[k] + 32'(ap_r[k]);
    for (int k = 0; k < NQ; k++)
      pf.w_q[k] = sq[k] + ((k == 4) ? 32'(af_r) : 32'(aq_r[k < 4 ? k : 4]));
    pf.h = h_ev;
    pf.h_v = sh;
  end

`ifndef SYNTHESIS
  // ---- +trace: the trace lines (opentpu/profile.py), from the events above -- exactly what the
  // board's trace buffer records (otpu_trace.sv, opentpu/hwtrace.py)
  bit trace;
  initial trace = $test$plusargs("trace");
  // the MXU's scale reads (otpu_mxu xs_*; rtlsim stats["mxu_scale"]): port A line requests, and
  // those that fetched a row's Sw alone
  always @(posedge clk)
    if (dump) $display("MXSCALE s%0d areq=%0d sw=%0d", SID, u_mxu.xs_a, u_mxu.xs_sw);
  // the ACT RAM's reads (otpu_mxu xa_*; rtlsim stats["act_reads"]): cycles, column reads, pops,
  // the columns of output rows at the pops, and the columns that ever read (a mask)
  always @(posedge clk)
    if (dump) $display("MXACT s%0d cyc=%0d rd=%0d pop=%0d use=%0d cols=%0d any=%0d", SID,
                       u_mxu.xa_cyc, u_mxu.xa_rd, u_mxu.xa_pop, u_mxu.xa_use, MCOLS,
                       64'(u_mxu.xa_any));
  always_ff @(posedge clk)
    if (trace) begin
      if (pf.sq.g) $display("T%0d G c=%0d s=%0d", SID, pf.sq.cyc, pf.sq.g_slot);
      for (int i = 0; i < 32; i++)
        if (pf.sq.e[i]) $display("T%0d E c=%0d s=%0d", SID, pf.sq.cyc, i);
      for (int u = 0; u < NUNITS; u++)
        if (pf.sq.s[u])
          $display("T%0d S c=%0d s=%0d u=%0d r=%0d", SID, pf.sq.cyc, pf.sq.s_slot[u], u,
                   pf.sq.s_rdy[u]);
      if (pf.sq.d)
        $display("T%0d D c=%0d s=%0d pc=%0d op=%02h w1=%08h w2=%08h w3=%08h", SID, pf.sq.cyc,
                 pf.sq.d_slot, pf.sq.d_pc, pf.sq.d_op, pf.sq.d_w1, pf.sq.d_w2, pf.sq.d_w3);
      if (pf.u_mxu)
        $display("T%0d U c=%0d u=1 starve=%0d bp=%0d frz=%0d deny=%0d", SID, pf.sq.cyc,
                 pf.u_mxu_v[0], pf.u_mxu_v[1], pf.u_mxu_v[2], pf.u_mxu_v[3]);
      if (pf.u_q) $display("T%0d U c=%0d u=2 frz=%0d", SID, pf.sq.cyc, pf.u_q_frz);
      if (pf.u_vpu) $display("T%0d U c=%0d u=3 frz=%0d", SID, pf.sq.cyc, pf.u_vpu_frz);
      if (pf.h)
        $display("T%0d H c=%0d bmxu=%0d bdma=%0d amxu=%0d aq=%0d", SID, pf.sq.cyc, pf.h_v[0],
                 pf.h_v[1], pf.h_v[2], pf.h_v[3]);
      if (pf.w) begin
        $display("T%0d P c=%0d n=%0d bm=%0d bd=%0d am=%0d aq=%0d mx=%0d fm=%0d fq=%0d fv=%0d fc=%0d",
                 SID, pf.sq.cyc, pf.w_n, pf.w_p[0], pf.w_p[1], pf.w_p[2], pf.w_p[3], pf.w_p[4],
                 pf.w_p[5], pf.w_p[6], pf.w_p[7], pf.w_p[8]);
        $display("T%0d Q c=%0d n=%0d bs=%0d as=%0d ms=%0d mb=%0d ff=%0d ld=%0d", SID, pf.sq.cyc,
                 pf.w_n, pf.w_q[0], pf.w_q[1], pf.w_q[2], pf.w_q[3], pf.w_q[4], pf.w_q[5]);
      end
    end
`endif
endmodule
