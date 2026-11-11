// VPU: LANES fp32 elements per cycle along a row of a [rows, cols] TMEM tile. Operand A from
// read port A, operand B from read port B (full tile, per-column), one broadcast word (per-row)
// or the scalar immediate (docs/isa.md, VOP).
//
// Pipelined for the FPGA clock. Split lanes: lanes 0..CL-1 are chains of NSLOT multiply-add
// slots (slot = (a*b)+c with its own input register, SL = 1 + LM + LA cycles; slot 1 has three
// more for the EXP2 range reduction); lanes CL..LANES-1 have slot 0 only. The composite
// functions (EXP2, RECIP, RSQRT, LOG2) are issued CL columns per cycle onto the long lanes, every
// other function LANES columns per cycle. Composite functions
// are slot programs, exactly the fp_add/fp_mul sequences of the ISA:
//   ADD/SUB/RSUB/MUL/OUTER  1 slot  EXP2/EXP2SUB 9 (sub, range reduction, 7 Horner steps)
//   RECIP             6 (2 Newton steps, a correction step)
//   RSQRT             10 (h = x/2, 2 Newton steps, a correction step)
//   LOG2              10 (t = m - 1, 8 Horner steps, + i2f(e) from slot 1's input stages)
//   MAX/MIN/COPY/ABS/FILL 0 (written the cycle after the read)
// Slot 0 is a*b + c*e (two products): e = 1.0 except for OUTER, dst*D(c) + B(r)*C(c). OUTER
// first reads its column vectors C (port B) and D (port A) into per-lane buffers, one chunk
// per cycle (at least two cycles, so a buffer word is written before it is read); the main
// pass then reads dst on port A, B(r) on port B lane 0, and C(c), D(c) from the buffers.
// The result is written from the slot where the function ends, so latency depends on the
// function. Several elementwise instructions overlap in the pipeline: every entry carries its
// function, and an instruction may start while others are in flight only if its latency is
// at least theirs, so writes and completions stay in start order. (The sequencer makes a VOP
// wait for the VOPs it depends on to complete.) Reductions run alone.
//
// Reductions: RMAX folds each chunk with a max tree (the order is total, so any order is
// exact). ARGMAX is RMAX with the index: the tree carries each value's lane (ties: the lower),
// the running maximum its column (ties: the older), and a row's (max, i2f(column + imm)) pair
// is written through lanes 0 and 1 four cycles after the row (the index add and the int ->
// fp32 conversion, pipelined; docs/isa.md). RSUM/RSSQ/RDOT implement isum_64 (of A, A*A, A*B): lane l owns the partials l,
// l+LANES, ... and adds each chunk's term into the partial last updated RL = 64/LANES chunks
// ago -- a loop of exactly RL cycles through a pipelined adder. Rows are padded with +0 terms
// to a multiple of 64 columns; the final partials of a row come out LANES per cycle, in order,
// and a streaming folding tree sums them while the next row streams: the levels across rows on
// LANES adders fed by delay lines on a fixed schedule, the levels across lanes on a pipeline
// of adders (level 2 reuses level 1's, which are idle then).
//
// RDOT (up to 256 rows into consecutive words) keeps its row sums in a buffer and writes them
// LANES per cycle after the last row: an RDOT then requests no TMEM writes while it streams,
// so it runs at full rate beside a DMA LD, whose writes take every bank each cycle (and come
// first). The other reductions write each row sum as it comes out.
//
// Everything (reads, lane pipelines, the tree, writes) advances on cycles the TMEM grant is
// given; nothing stalls. With WBUF (the board) the grant only takes the head of a two-entry
// write buffer, and everything else advances on a registered enable: there is room in the
// buffer for the cycle's writes.
//
// With WBUF the VPU is a front and a back end (FB). The front (the command state, the issue,
// the read requests, mi, mi2) advances on en, which a lost TMEM read (rhold, decided in TMEM's
// arbitration that same cycle) holds the next cycle. The back end (m0 and the TMEM data, the
// lanes, the trees, the composites, the tail, the writes and the completions: ~21k registers
// on the board) is the same pipeline a cycle later: it advances on en_b, en a cycle late (a
// flip-flop: its copies have no logic in front of them, only route), and takes mi2 and the read
// data through registers loaded every cycle (mi2_d, ta_d, tb_d), which also carry the TMEM's
// data a cycle towards the lanes. Every hold of the front is then the back end's a cycle later,
// so the back end's schedule (the reductions' partial loop, the composites' loop) is exactly
// today's; what the back end reads of the front's state it reads a cycle late too (the *_b
// copies). Its completions reach the front a cycle later (ew_n, red_act: a begin may wait a
// cycle), the write buffer's room is decided a cycle further ahead (more slots), and the
// composites' issue hold looks one step further when the back end steps (hold_n). In stream
// mode the front issues nothing and en_b is the stream's enable, as before (en_q).
//
// Stream engine (HAS_SE, LANES = 8; docs/stream.md): the VPU is SE's front. A stream (the DMA's
// DSTEP or STREAM) asks for SE with ss_req: rdy falls, the VOPs queued and in flight finish,
// then ss_gnt rises and holds until ss_req falls. While it is up SE issues nothing and makes no
// TMEM access; everything advances on the stream's pe; slot 0 of every lane runs the RDOT
// partial loop on the tail's X segment (xd) and dot-A vector (xa), and u_vt folds it into the
// row's kv for the tail (otpu_se_tail: the d stage, the delay line, U and Q). Every ss_* input
// is registered twice here (the DMA registers its side too), so SE runs two cycles behind the
// DMA's pe: the enable (~30K loads) is then a flip-flop copy of one computed a cycle ahead
// (en_q), with no logic in front of its replicas; ss_y_v and ss_o_v are qualified with SE's pe
// (a Y / an O this cycle).
//
// With the stream engine (docs/stream.md 11):
//   one tree  the tail's dot Q folds on u_vt too, in windows between dot A's (the tail pads its
//             Q path so they never meet); a FIFO of the rows' kinds routes each root
//   COMP8     no long lanes: the composites (EXP2, EXP2SUB, RECIP, RSQRT, LOG2) are issued
//             LANES columns per cycle into otpu_se_comp, which loops each chunk through slot 0
//             and the tail's U and Q (its stages 0, 1, 2) and hands the result back in start
//             order; a composite chunk is not issued while comp's `hold` says slot 0 will be
//             taken when it gets there. The latency rule is unchanged: a composite's latency
//             (passes x 25 cycles; CNS 5: two more stages of comp's own, 2 passes x 39)
//             orders the functions as their slot counts do (at CNS 5 RECIP's equals EXP2's: the
//             rule only waits more).
// Without it (HAS_SE = 0 or LANES != 8) the composites run on the long lanes' slot chains.
//
// FOLLOW (otpu_seq FOL: wait entries at S > 1, docs/isa.md "Wait counts"): a VOP started with
// fol (a reduction of one row's A) may run while the GATHER it reads is still writing: its A
// chunks are read only below fol_top, the top of that GATHER's writes into this slice's TMEM
// (otpu_slice; the GATHER writes upwards from its dst), until fol_done (the collective
// completes). The whole VPU waits for them as for the grant (f_ok in the enable: a reduction's
// partial loop takes no bubble), and the instruction begins only once its first chunks are
// written (f_wait), so the enable's lag (up to three cycles: SE's en_q) never reads ahead.
//
// FSUM (C8 only; Config.FSUM, docs/isa.md VOP): an EXP2SUB with flag FSUM also writes its rows'
// isum_64 at T[imm + r], RSUM's sums of its results bit for bit, by the time an RSUM with flag
// FSUM after it (its take) is done. Each lane adds the results it
// hands back (otpu_se_comp's output, the chunk's meta: its index in the row mod 8, first, the
// row's last, the row's parity) into its partials (partial (c mod 8) * 8 + l, as RSUM's loop)
// on an adder of its own, two rows' 8 partials a lane (the row being summed, the row draining).
// A row's final partials go into u_vt LANES a cycle, 8 cycles (sub 0..7), rows a multiple of 8
// cycles apart (a phase counter), the roots into RDOT's row-sum buffer, flushed after the last
// row (in cycles with no result to write). The EXP2SUB is done at its last result, as any
// elementwise VOP, so the next ones (and its results' readers) need not wait for the sums; while
// they are pending (fs_act) no reduction, FSUM, take or stream begins (u_vt and the buffer are
// the sums'), and a take begins only after them: it writes nothing and completes as an empty
// VOP, its footprint (RSUM's) the sums' fence for the sequencer. cols >= 64 (every partial
// gets a first term; rows end >= 8 chunks apart, so a row drains before the row after the next
// one reaches its bank) and rows <= 256 (the buffer); B not a scalar (imm is the address).
//
// VSIG (C8; Config.VSIG, docs/isa.md VOP SIGMOID / SILU): SIGMOID's recip(exp2(A*B) + 1) and
// SILU's A times it, binary composites (B in any mode: the MUL's factor) on otpu_se_comp in four
// passes at CNS 5 and six at CNS 3 (its header), bit for bit the VOPs MUL, EXP2, ADD 1.0, RECIP
// (MUL). Their slot count (15) is the latency rule's longest, as their passes are at either CNS.
//
// OUTER = 0 builds the VPU without VOP OUTER (Config.OUTER off: a target whose DeltaNet runs
// DSTEP, the WG targets): slot 0 is a*b + c (e is 1.0 for every other function, the composites' stage 0
// and the stream's dot A, and add(p, mul(c, 1)) == add(p, c) bit for bit: the adder flushes c and
// returns NaN for any NaN), OUTER's fill and column buffers are not built, and an OUTER (the
// compiler and the ISA simulator refuse one) is a simulation error.
//
// MAC (Config.VMAC; docs/isa.md VOP MAC): dst = add(dst, mul(A, B)) in place, B per column
// (B_COL, cols <= 256) or the immediate (B_SCALAR), the MUL then ADD of `dst + A * B` bit for bit
// in one slot-0 pass (a*b + c: A, B(c), dst). B_COL first reads B into the column buffer as
// OUTER's fill reads C (port B, >= 2 chunks, port A idle); the main pass then reads A on port A
// and dst on port B, a full tile at row stride drs (B_FULL's row walk: b_row from dst). Its
// fill, the C buffer and the operand case are built only with MAC (independent of OUTER: a VPU
// with MAC and without OUTER has the fill and C buffer, not D's); without it a MAC is a
// simulation error.
//
// OUTER1 (Config.OUTER1, MAC 1 and OUTER 0; docs/isa.md VOP OUTER): VOP OUTER with decay one
// (flag DONE) alone, on MAC's datapath: the fill reads C into MAC's C buffer (port B, port A
// idle, as DONE's fill), the main pass dst on port A and B(r) on port B lane 0 (OUTER's reads),
// and slot 0's a*b + c takes B(r), C(c), dst: add(mul(B(r), C(c)), dst), which is add(mul(dst,
// 1.0), mul(B(r), C(c))) bit for bit (mul(x, 1.0) flushes x as the adder flushes c, and the add
// commutes). No D buffer, no second product; an OUTER with a decay word is a simulation error.
module otpu_vpu
  import otpu_pkg::*;
  import otpu_fp::*;
#(
  parameter int LANES = 8,
  parameter int CL    = (LANES >= 8) ? LANES / 4 : 1,   // lanes with the composite functions
  parameter int SID   = 0,
  // WBUF: the TMEM grant only takes writes from a two-entry write buffer, and the pipeline
  // advances on a registered enable (buffer room) -- for arbiters whose grant depends only on
  // the VPU's writes (the board's: every read port has its own TMEM copy). 0: everything
  // advances on the grant itself.
  parameter bit WBUF  = 0,
  parameter bit HAS_SE = 1'b0,                        // the stream engine (LANES = 8)
  // the DMA's streams run on it (0: on otpu_se_eng beside the VPU, otpu_slice SE_R > 0; the
  // engine's tail runs the composites alone, in VOP mode always: no grant, ss_act's copies
  // constants, so synthesis trims the stream-only half)
  parameter bit SE_SS = 1'b1,
  parameter bit OUTER = 1'b1,                         // VOP OUTER (0: none, see the header)
  parameter bit DSTEP2 = 1'b0,                        // DSTEP's pair (otpu_se_tail)
  parameter bit FOLLOW = 1'b0,                        // a VOP reads behind a GATHER (fol_*)
  // C8: otpu_se_comp's stages per lane: 3 (slot 0, the tail's U and Q), or 5 (two more of its
  // own per lane: EXP2 / EXP2SUB / RSQRT / LOG2 in two passes, 4 columns a cycle; bit-exact)
  parameter int CNS = 3,
  parameter bit FSUM = 1'b0,                          // EXP2SUB's fused row sums (C8; header)
  parameter bit VSIG = 1'b0,                          // SIGMOID, SILU (C8; header)
  parameter bit MAC = 1'b0,                           // VOP MAC (header)
  parameter bit OUTER1 = 1'b0                         // VOP OUTER decay one on MAC's (header)
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    start,
  input  cmd_t                    cmd,
  output logic                    rdy,
  output logic                    done,
  input  logic                    gnt,        // TMEM grant: hold everything when low (WBUF:
                                               // the head write is taken)
  output logic                    ren,        // the lanes take the read data (WBUF: the
                                               // registered enable; else the grant)
  input  logic                    fol,        // FOLLOW: the start follows a running GATHER
  input  logic [31:0]             fol_top,    // ... the top of its writes into TMEM
  input  logic                    fol_done,   // ... it completed (pulse)
  input  logic                    rhold,      // WBUF: a read of this cycle lost its TMEM RAM
                                               // (otpu_tmem RPL): hold the next cycle while it
                                               // is read again
  output logic [LANES-1:0]        ta_en,
  output logic [LANES-1:0][31:0]  ta_addr,
  input  logic [LANES-1:0][31:0]  ta_data,
  output logic [LANES-1:0]        tb_en,
  output logic [LANES-1:0][31:0]  tb_addr,
  input  logic [LANES-1:0][31:0]  tb_data,
  // ta_* / tb_* by TMEM bank (otpu_tmem PBR: the ports' rotator merge, registered with them)
  output logic [LANES-1:0]        ta_ben, tb_ben,
  output logic [LANES-1:0][31:0]  ta_bad, tb_bad,
  output logic [LANES-1:0]        tw_en,
  output logic [LANES-1:0][31:0]  tw_addr,
  output logic [LANES-1:0][31:0]  tw_data,
  // profiling: a cycle after an instruction ends, its cycles frozen by the TMEM grant
  output logic                    pf_u,
  output logic [31:0]             pf_frz,
  // stream engine (the DMA's side, registered there)
  input  logic                    ss_req,     // a stream holds SE (level, until its last O)
  output logic                    ss_gnt,     // SE is idle of VOPs and in stream mode (level)
  input  ss_cfg_t                 ss_cfg,     // valid while ss_req
  input  logic                    ss_pe,      // advance
  input  logic                    ss_in_v,
  input  f32_t                    ss_in_d [LANES],
  input  logic [2:0]              ss_fk,      // fills: kind (SF_*), index, data
  input  logic [4:0]              ss_fi,
  input  f32_t                    ss_fd [LANES],
  output logic                    ss_y_v,     // an updated segment this cycle
  output f32_t                    ss_y_d [LANES],
  output logic                    ss_o_v,     // a row's O this cycle
  output f32_t                    ss_o_d
);
  localparam int LM = 2, LA = 4;
  localparam int SL = 1 + LM + LA;          // cycles per slot
  localparam int NSLOT = 10;
  localparam int NP = 64;                    // RSUM/RSSQ partials (isum_64)
  localparam int RL = NP / LANES;            // partial loop length in chunks
  localparam int LW = $clog2(LANES);
  // the stream engine: its tail is built for 8 lanes (slot 0's partial loop, RL = 8)
  localparam bit SE = HAS_SE && (LANES == 8);
  localparam bit C8 = SE;                    // the composites on otpu_se_comp
  localparam int NCL = C8 ? LANES : (CL < LANES) ? CL : LANES;   // composite columns per cycle
  localparam int NLL = C8 ? 0 : NCL;         // long lanes (the composites' slot chains)
  localparam int NSX = C8 ? 1 : NSLOT;       // slots along the lane taps
  localparam int CLW = $clog2(NCL);
  // TMEM word address bits (the TMEM has far fewer words and $fatals past them); the ports
  // zero-extend to 32 bits
  localparam int AW = 20;
  // OUTER's column buffers: 256 columns (docs/isa.md) over the lanes
  localparam int CBD = 256 / LANES;
  localparam int CBW = $clog2(CBD);
  initial if (RL < LA || NP % LANES != 0)
    $fatal(1, "otpu_vpu: LANES must be a power of two <= 16");
  initial if (OUTER1 && (OUTER || !MAC))
    $fatal(1, "otpu_vpu: OUTER1 is OUTER's decay one on MAC's datapath (MAC 1, OUTER 0)");
  localparam bit OXV = OUTER || OUTER1;      // VOP OUTER issues (full, or decay one)
  localparam int TA = 1 + LM + LA;           // stream: the tail's X registers -> slot 0's pacc

  localparam f32_t F_NZ = 32'h8000_0000;     // -0: (a*b) + -0 == a*b exactly

  // ------------------------------------------------------------------ command state
  logic        issuing, red_act;
  logic [AW-1:0] dst, a, b;
  logic [31:0] imm;
  logic [15:0] rows, cols, drs, ars, brs;
  logic [7:0]  func;
  logic [1:0]  bmode;
  logic [15:0] ir, ic, nch, ch;               // issue: row, column, chunks per row, chunk
  logic [AW-1:0] a_row, b_row, d_row;         // row base addresses (no multipliers)
  // OUTER: the fill phase (ch counts its chunks up to nfill), the C and D addresses, the flags
  logic        filling;
  logic [15:0] nfill;
  logic [AW-1:0] oc_a, od_a;
  logic        od_sc, od_one;
  logic        mac_i;                          // MAC: the instruction issuing is a MAC
  logic [3:0]  nslots;                         // slots of this function
  logic [7:0]  tag;                            // instruction tag
  // FSUM: an EXP2SUB with its row sums began and its sums are not written (see the header);
  // fs_b the back end's (set at its begin, a cycle late with FB; cleared at the flush's end);
  // fs_i: the instruction issuing is that EXP2SUB; fs_rows, fs_imm: its rows and sums' address
  // (the VOPs after it begin before the flush)
  logic        fs_act, fs_b, fs_i;
  logic [15:0] fs_rows;
  logic [31:0] fs_imm;
  // cycles an instruction was frozen by the TMEM grant, for the profiler: the condition is
  // registered (st_c) and summed a cycle late, off the grant path; the count is st_frz + st_c
  // (an instruction begins on a granted cycle, so a begin never drops a pending frozen cycle
  // of its own)
  logic [31:0] st_frz;
  logic        st_c;
  // FB (WBUF): the front / back split (see the header). en_b: the back end's enable (en a cycle
  // late; en_l are its copies); the front's state as the back end reads it, a cycle late (set
  // in `sequencing`); red_act_b: the back end's red_act
  localparam bit FB = WBUF;
  logic          en_b;
  logic          red_act_b;
  logic [7:0]    func_b, tag_b;
  logic [31:0]   imm_b, st_frz_b;
  logic [15:0]   rows_b, drs_b;
  logic [AW-1:0] dst_b;
  logic          od_one_b, od_sc_b, st_c_b;
  wire is_sum_b = (func_b == V_RSUM) || (func_b == V_RSSQ) || (func_b == V_RDOT);
  wire is_am_b = (func_b == V_ARGMAX);
  wire is_mx_b = (func_b == V_RMAX) || is_am_b;

  function automatic logic [3:0] n_slots(input logic [7:0] f);
    case (f)
      V_ADD, V_SUB, V_RSUB, V_MUL, V_OUTER: return 4'd1;
      V_EXP2, V_EXP2SUB:           return 4'd9;
      V_RECIP:                     return 4'd6;
      V_RSQRT, V_LOG2:             return 4'd10;
      V_SIGMOID, V_SILU:           return 4'd15;    // (VSIG: four or six passes, the longest)
      default:                     return (MAC && f == V_MAC) ? 4'd1 : 4'd0;
    endcase
  endfunction

  function automatic logic comp_f(input logic [7:0] f);
    return n_slots(f) > 1;
  endfunction

  // the classes of functions the lane taps after slot 0 tell apart (slot 0 reads m0's func)
  localparam logic [2:0] C_OTH = 3'd0, C_ONE = 3'd1, C_EXP = 3'd2, C_RCP = 3'd3, C_RSQ = 3'd4,
                         C_LOG = 3'd5;
  function automatic logic [2:0] f_cls(input logic [7:0] f);
    case (f)
      V_ADD, V_SUB, V_RSUB, V_MUL, V_OUTER: return C_ONE;
      V_EXP2, V_EXP2SUB:           return C_EXP;
      V_RECIP:                     return C_RCP;
      V_RSQRT:                     return C_RSQ;
      V_LOG2:                      return C_LOG;
      default:                     return (MAC && f == V_MAC) ? C_ONE : C_OTH;  // (reductions)
    endcase
  endfunction
  // a function of the class: the same as each of its functions at every slot s >= 1 (EXP2SUB
  // differs from EXP2 at slot 0 only); C_OTH's has no slots
  function automatic logic [7:0] cls_f(input logic [2:0] c);
    case (c)
      C_ONE:   return V_ADD;
      C_EXP:   return V_EXP2;
      C_RCP:   return V_RECIP;
      C_RSQ:   return V_RSQRT;
      C_LOG:   return V_LOG2;
      default: return V_COPY;
    endcase
  endfunction

  // the functions that read B in its bmode (otpu_pkg's reads_b; MAC's only with MAC, so the read
  // enables' function without it is the one before MAC)
  function automatic logic vreads_b(input logic [7:0] f);
    return is_binary(f) || f == V_RDOT || f == V_OUTER || (MAC && f == V_MAC);
  endfunction

  // TMEM word address base + c + l, AW bits, zero-extended to the port
  function automatic logic [31:0] tm_addr(input logic [AW-1:0] base, input logic [15:0] c,
                                          input int l);
    logic [AW-1:0] t;
    t = base + AW'(c) + AW'(l);
    return 32'(t);
  endfunction
  // composite columns a chunk: NCL; in simulation +vt_cw=W issues W (a composite path W lanes
  // wide at today's latency: docs/seq_vpu.md, EXPERIMENTAL)
  logic [15:0] cwid;
`ifndef SYNTHESIS
  int vt_cw = NCL;
  initial void'($value$plusargs("vt_cw=%d", vt_cw));
  assign cwid = 16'(vt_cw);
`else
  assign cwid = 16'(NCL);
`endif
  wire [15:0] iwid = comp_f(func) ? cwid : 16'(LANES);   // columns issued per cycle

  // C8: otpu_se_comp, whose stages are slot 0 (s = 0) and the tail's U (1) and Q (2), and at
  // CNS 5 two of its own (3, 4). A composite chunk is not issued while comp says slot 0 is taken
  // when it would get there (hold: HA = 2 cycles ahead, the read -> mi -> m0 path)
  initial if (CNS != 3 && CNS != 5) $fatal(1, "otpu_vpu: CNS %0d (3 or 5)", CNS);
  initial if (FSUM && !C8) $fatal(1, "otpu_vpu: FSUM needs the composites on otpu_se_comp (C8)");
  initial if (VSIG && !C8) $fatal(1, "otpu_vpu: VSIG needs the composites on otpu_se_comp (C8)");
  logic             c_hold, c_hold_n, c_ov, c_s0;
  logic [CNS-1:0]   c_sel;
  f32_t             c_sa [CNS][LANES], c_sb [CNS][LANES], c_sc [CNS][LANES], c_se [CNS][LANES];
  f32_t             c_s0y [LANES], c_uy [LANES], c_qy [LANES];   // the stages' results
  f32_t             c_ia [LANES], c_ib [LANES], c_od [LANES];
  logic [LANES-1:0] c_om;
  // {waddr, all_last}; FSUM: {fs, first, row_last, parity, sub[2:0], waddr, all_last}
  localparam int FMW = (FSUM && C8) ? 7 : 0;
  logic [AW+FMW:0]  c_ometa;
  // EXPERIMENTAL (docs/seq_vpu.md), simulation only: a narrower vector path at today's latency.
  // +vt_ew=N / +vt_comp=N: an elementwise / composite chunk issues on one enabled cycle in N
  // (a path 1/N as wide; for the composites +vt_cw above is the exact model, this a gap at
  // issue that the composite loop's schedule may partly absorb); +vt_red=N: while a reduction
  // runs, the VPU advances on one cycle in N (its bubbles would break the partial loop's
  // schedule, so the whole VPU waits). Defaults 1.
  logic             iss_thr;                   // the chunk may issue this cycle
  logic             red_thr;                   // the VPU may advance this cycle (a reduction's)
  logic             f_ok;                      // FOLLOW: ... its reads are below fol_top
  logic             f_wait;                    // FOLLOW: the head may not begin yet
`ifndef SYNTHESIS
  int vt_ew = 1, vt_comp = 1, vt_red = 1;
  int vt_c = 0, vt_r = 0;
  initial begin
    void'($value$plusargs("vt_ew=%d", vt_ew));
    void'($value$plusargs("vt_comp=%d", vt_comp));
    void'($value$plusargs("vt_red=%d", vt_red));
  end
  always_comb begin
    int n;
    n = is_reduce(func) ? 1 : comp_f(func) ? vt_comp : vt_ew;
    iss_thr = (n <= 1) || (vt_c % n == 0) || filling;
    red_thr = (vt_red <= 1) || !red_act || (vt_r % vt_red == 0);
  end
  always @(posedge clk) begin
    if (en && issuing) vt_c <= vt_c + 1;
    if (red_act) vt_r <= vt_r + 1;
  end
`else
  assign iss_thr = 1'b1;
  assign red_thr = 1'b1;
`endif
  // FB: comp runs on en_b, a step behind the front when the back end steps this cycle; its
  // schedule as the front would see it is then a tap further on (hold_n)
  wire              c_hold_f = (FB && en_b) ? c_hold_n : c_hold;
  wire              iss_ok = !(C8 && comp_f(func) && c_hold_f) && iss_thr;

  wire is_sum = (func == V_RSUM) || (func == V_RSSQ) || (func == V_RDOT);
  wire is_am = (func == V_ARGMAX);
  wire is_mx = (func == V_RMAX) || is_am;
  wire is_red = is_sum || is_mx;

  function automatic logic red_f(input logic [7:0] f);
    return is_reduce(f);
  endfunction

  // command queue (the sequencer starts instructions into it)
  cmd_t        cq [2];
  logic [1:0]  cq_n;                          // queued
  logic        cq_h;
  // elementwise instructions issued, not done: one may start every other cycle, so up to
  // latency / 2 are in flight (COMP8's RSQRT and LOG2: 100 cycles)
  logic [6:0]  ew_n;
  logic [3:0]  last_tap;                      // slots of the last one started
  // stream mode (ss_gnt); SE takes no VOP while a stream asks for it (ss_rq: ss_req registered,
  // so no path runs from the DMA through rdy into the sequencer)
  (* max_fanout = 64 *) logic ss_act;
  logic ss_rq;
  assign rdy = (cq_n < 2) && !(SE && ss_rq);
  // the stream: the ss_* inputs registered (sen: its pe; s_init: its first granted cycle), the
  // tail's X registers (s_xd, s_xa, s_xm), and s_xm at slot 0's pacc (s_mt)
  (* max_fanout = 64 *) logic sen;
  logic       ss_act_d;                        // ss_act's next value
  // ss_act's copies, three per lane (see g_sar): the lane's RDOT term (sa_r), the tail's U
  // operand registers and Q multiplier operands (cm_u, cm_q: its complement)
  logic [LANES-1:0] sa_r, cm_u, cm_q;
  logic       s_init, s_in_v, s_first;
  f32_t       s_in_d [LANES], s_fd [LANES], s_xd [LANES], s_xa [LANES];
  logic [2:0] s_fk;
  logic [4:0] s_fi [LANES];
  ss_cfg_t    s_cfg;
  ss_meta_t   s_xm, s_mt;
  // the tail's Q final partials, registered here (q_*); the root is a Q dot's (vt_q)
  logic       q_cap, q_rl, vt_q;
  logic [7:0] q_sub;
  f32_t       q_pd [LANES];
  cmd_t hc;
  assign hc = cq[cq_h];
  wire  [7:0] hf = hc.w6[23:16];
  // the head is an EXP2SUB with its row sums (FSUM), or an RSUM that takes them (no work: it
  // completes as an empty instruction does, once they are written)
  wire  h_fs = FSUM && C8 && hf == V_EXP2SUB && hc.flags[VF_FSUM];
  wire  h_take = FSUM && C8 && hf == V_RSUM && hc.flags[VF_FSUM];
  wire  h_empty = (hc.w4[15:0] == 0) || (hc.w4[31:16] == 0) || h_take;
  wire  can_begin = (cq_n != 0) && !issuing && !red_act &&
                    !(fs_act && (red_f(hf) || h_fs)) &&
                    ((red_f(hf) || h_empty) ? (ew_n == 0) :
                                              (ew_n == 0 || n_slots(hf) >= last_tap));
  wire  busy = issuing || red_act || ew_n != 0 || fs_act;


  // the pipeline enable: the grant, or (WBUF) the write buffer's room, registered. The enable
  // reaches every pipeline register of the lanes and the tree (~21k loads on the board), so it
  // must not come from the arbiter in the same cycle.
  logic en;
  // ... and its copies, one per lane (g_wbuf's g_enl): the lanes' stages, the tail's units and the
  // composites' lanes take their lane's, so no one register's replicas cross the VPU
  logic [LANES-1:0] en_l;
  assign ren = en;

  // ------------------------------------------------------------------ issue
  // The read addresses follow the mode alone (OUTER fill or not, B's mode): a lane's address
  // only counts while its enable is up, and a rotator port takes lane 0's whenever any lane
  // is, so the lane tests (ic + l < cols, the issue checks) reach the enables but not the
  // addresses, and TMEM's bank addresses no longer wait for them (133.33 MHz, fused-133
  // c2830d6: cols -> the lane test -> tb_en -> the address multiplexer -> port 4's bank rows
  // -> block RAM ADDRB, 9-10 levels, +0.130 ns).
  // TMEM's input stage (release r2, docs/tmem_nocopy.md 7): the request (ta_/tb_*_c) goes to
  // TMEM from flip-flops, a cycle after it is formed, advancing with the enable: the chunk's
  // address arithmetic, its lane tests and the issue checks end at the register, not at the
  // TMEM's ranks and pumped banks (r2m4: cols / ic -> the request address -> the pump banks'
  // read pins, clk -> clk2x, 55% of its TNS). The chunk (mi) waits a cycle in mi2 for it.
  // The request by bank too (ta_ben / tb_ben, otpu_tmem PBR), so TMEM's ranks and compares start
  // at flip-flops, not at its rotation of the lanes (r150d, the m8 card at a real 150 MHz:
  // tb_en -> the rotation -> the ranks -> a pumped bank's address, 7 levels, 5.2 ns of route)
  logic [LANES-1:0]       ta_en_c, tb_en_c, ta_ben_c, tb_ben_c;
  logic [LANES-1:0][31:0] ta_addr_c, tb_addr_c, ta_bad_c, tb_bad_c;
  otpu_tmem_breq #(.LANES(LANES)) u_tab (.en(ta_en_c), .a0(ta_addr_c[0]), .ben(ta_ben_c), .bad(ta_bad_c));
  otpu_tmem_breq #(.LANES(LANES)) u_tbb (.en(tb_en_c), .a0(tb_addr_c[0]), .ben(tb_ben_c), .bad(tb_bad_c));
  always_ff @(posedge clk)
    if (rst) begin ta_en <= '0; tb_en <= '0; ta_ben <= '0; tb_ben <= '0; end
    else if (en) begin
      ta_en <= ta_en_c; tb_en <= tb_en_c; ta_addr <= ta_addr_c; tb_addr <= tb_addr_c;
      ta_ben <= ta_ben_c; tb_ben <= tb_ben_c; ta_bad <= ta_bad_c; tb_bad <= tb_bad_c;
    end
  logic [LANES-1:0] imask;
  logic             irow_last, iall_last;
  always_comb begin
    ta_en_c = '0; tb_en_c = '0;
    imask = '0;
    irow_last = (ch + 1 == nch);
    iall_last = irow_last && (ir + 1 == rows);
    for (int l = 0; l < LANES; l++) begin
      if (filling) begin
        ta_addr_c[l] = od_sc ? 32'(od_a) : tm_addr(od_a, ic, l);
        tb_addr_c[l] = tm_addr(oc_a, ic, l);
      end else begin
        ta_addr_c[l] = tm_addr(a_row, ic, l);
        tb_addr_c[l] = (bmode == B_COL && !mac_i) ? tm_addr(b, ic, l) :   // (MAC: dst's row)
                     (bmode == B_ROW && l == 0) ? 32'(b_row) : tm_addr(b_row, ic, l);
      end
    end
    if (issuing && filling) begin
      // OUTER fill: C on port B; D on port A (one word on lane 0 for DSCALAR, none for DONE);
      // MAC's: B on port B (od_one)
      for (int l = 0; l < LANES; l++) begin
        if (32'(ic) + 32'(l) < 32'(cols)) begin
          tb_en_c[l] = 1'b1;
          if (!od_one && (!od_sc || l == 0)) ta_en_c[l] = 1'b1;
        end
      end
    end else if (issuing && iss_ok) begin
      for (int l = 0; l < LANES; l++) begin
        if (32'(ic) + 32'(l) < 32'(cols) && 16'(l) < iwid) begin
          imask[l] = 1'b1;
          ta_en_c[l] = (func != V_FILL);
          if (mac_i || (vreads_b(func) && (bmode == B_FULL || bmode == B_COL))) tb_en_c[l] = 1'b1;
        end
      end
      if (bmode == B_ROW && vreads_b(func) && imask[0]) tb_en_c[0] = 1'b1;
    end
  end
`ifndef SYNTHESIS
  // the checks start once the reset has held two edges (registers start arbitrary, and so does a
  // unit's reset register until the slice's reset pipeline fills: otpu_slice)
  bit rst_p, rst_seen;
  initial begin rst_p = 1'b0; rst_seen = 1'b0; end
  always @(posedge clk) begin rst_p <= rst; if (rst && rst_p) rst_seen <= 1'b1; end
  // an enabled lane's address is the one the lane tests chose
  always @(posedge clk)
    if (!rst && rst_seen)
      for (int l = 0; l < LANES; l++) begin
        logic [31:0] ra, rb;
        ra = filling ? (od_sc ? 32'(od_a) : tm_addr(od_a, ic, l)) : tm_addr(a_row, ic, l);
        rb = filling ? tm_addr(oc_a, ic, l) :
             (bmode == B_FULL || mac_i) ? tm_addr(b_row, ic, l) :
             (bmode == B_COL) ? tm_addr(b, ic, l) : 32'(b_row);
        if ((ta_en_c[l] && ta_addr_c[l] != ra) || (tb_en_c[l] && tb_addr_c[l] != rb))
          $fatal(1, "otpu_vpu: lane %0d's read address is not its mode's", l);
      end
`endif

  // meta of a chunk; m0 describes the data arriving this cycle (read last cycle)
  typedef struct packed {
    logic             v;
    logic [7:0]       tag;
    logic [7:0]       func;
    logic [1:0]       bmode;
    logic [31:0]      imm;
    logic [LANES-1:0] mask;
    logic [AW-1:0]    waddr;     // elementwise: dst row + column; reductions: dst row
    logic             row_last;  // last chunk of its row
    logic             all_last;  // last chunk of the instruction
    logic             first;     // reductions: chunk index < RL (partials start at +0)
    logic             final_;    // reductions: chunk index >= nch - RL (partials become final)
    logic [7:0]       sub;       // chunk index mod RL
    logic             sq;        // func == V_RSSQ (the squarer's operand select, from a flip-flop)
    logic             dot;       // func == V_RDOT (the product's B operand, from a flip-flop)
    logic             fill;      // an OUTER fill chunk (not written): into the column buffers
    logic [CBW-1:0]   cb;        // OUTER: the column-buffer word of the chunk (chunk in row)
    logic [15:0]      col;       // the chunk's first column (ARGMAX's index)
    logic             fs;        // FSUM: the chunk's results are summed (an EXP2SUB's, FSUM)
    logic             par;       // FSUM: its row's parity (the partials' bank)
  } meta_t;
  // mi: the chunk whose read request forms this cycle; mi2: that chunk with its request in the
  // request register (TMEM reads it then); m0: the chunk one granted cycle later, together with
  // its TMEM data registered (xa, xb), so no path runs from the TMEM block RAMs into the lanes'
  // multipliers in one cycle
  meta_t m0, mi, mi2;
  always_ff @(posedge clk) if (rst) mi2 <= '0; else if (en) mi2 <= mi;
  // FB: mi2 and the TMEM read data a cycle later, every cycle (no enable): the back end's input
  meta_t mi2_d;
  logic [LANES-1:0][31:0] ta_d, tb_d;
  if (FB) begin : g_fbd
    always_ff @(posedge clk) begin
      mi2_d <= rst ? '0 : mi2;
      ta_d <= ta_data;
      tb_d <= tb_data;
    end
  end else begin : g_nfbd
    assign mi2_d = mi2;
    assign ta_d = ta_data;
    assign tb_d = tb_data;
  end
  logic [LANES-1:0][31:0] xa, xb;
  f32_t xc [LANES], xd [LANES];               // OUTER: C(c), D(c) from the buffers, with xa/xb
  // stream mode: m0 is an RDOT with no chunk (slot 0 on the partial loop, fed by the tail)
  localparam meta_t M_SS = '{func: V_RDOT, default: '0};
  // m0's function decoded as it loads (m0d = fdec(m0.func)): the lanes' slot-0 operand and result
  // selects, tap 0's hit, the class and otpu_se_comp's entry come from flip-flops, not from
  // compares of m0.func (133.33 MHz, the 100 MHz fused placement: m0.func (fo 119) -> short lane
  // rc / rb, 13.6 / 13.8 levels, -1.2 ns; the tap-0 hit and the composite test fan out to every
  // lane's result and operand muxes)
  localparam int NVF = 24;                    // function codes V_ADD .. V_MAC
  typedef struct packed {
    logic [NVF-1:0] fh;                       // one-hot func (none for codes >= NVF)
    logic           s0;                       // ends at tap 0: no slots, not a reduction
    logic           cf;                       // a composite (comp_f)
    logic [2:0]     cc;                       // its class (f_cc: otpu_se_comp's entry)
    logic [2:0]     cls;                      // f_cls(func)
  } fdec_t;
  function automatic logic [NVF-1:0] foh(input logic [7:0] f);
    return NVF'(1) << f;
  endfunction
  function automatic fdec_t fdec(input logic [7:0] f);
    fdec_t d;
    d.fh = foh(f) & ~(NVF'(!OXV) << V_OUTER)      // (OUTER 0: its operand cases never taken)
                 & ~(NVF'(!MAC) << V_MAC);       // (MAC 0: nor MAC's)
    d.s0 = !red_f(f) && n_slots(f) == 4'd0;
    d.cf = comp_f(f);
    d.cc = f_cc(f);
    d.cls = f_cls(f);
    return d;
  endfunction
  fdec_t m0d;
  always_ff @(posedge clk)
    if (rst) begin
      m0 <= '0;
      m0d <= fdec(8'd0);
    end else if (en_b) begin
      m0 <= (SE && ss_act) ? M_SS : mi2_d;
      m0d <= fdec((SE && ss_act) ? M_SS.func : mi2_d.func);
      xa <= ta_d;
      xb <= tb_d;
    end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && rst_seen && m0d != fdec(m0.func)) $fatal(1, "otpu_vpu: m0d is not m0's func");
  always @(posedge clk)
    if (!OUTER && !rst && rst_seen && issuing && func == V_OUTER && !(OUTER1 && od_one))
      $fatal(1, "otpu_vpu: VOP OUTER on a VPU built without it (OUTER 0; OUTER1: decay one only)");
  always @(posedge clk)
    if (!VSIG && !rst && rst_seen && issuing && (func == V_SIGMOID || func == V_SILU))
      $fatal(1, "otpu_vpu: VOP SIGMOID / SILU on a VPU built without them (VSIG 0)");
  always @(posedge clk)
    if (!rst && rst_seen && issuing && func == V_MAC &&
        (!MAC || bmode == B_FULL || bmode == B_ROW || (bmode == B_COL && cols > 16'd256)))
      $fatal(1, "otpu_vpu: VOP MAC without MAC, or its B not per column (<= 256) nor a scalar");
`endif
  // OUTER's column buffers (LUT RAM): written from the fill chunks' data at m0, read with the
  // main chunks' data (mi.cb) -- the fill runs >= 2 chunks ahead, so a word is written first
  // (OUTER 0: neither written nor read, so synthesis leaves them out; with MAC, C's is MAC's B
  // and OUTER1's C, D's never read)
  for (genvar l = 0; l < LANES; l++) begin : g_cbuf
    f32_t cbuf [CBD], dbuf [CBD];
    always_ff @(posedge clk) if (en_l[l]) begin
      if (m0.fill) begin
        cbuf[m0.cb] <= xb[l];
        dbuf[m0.cb] <= od_one_b ? F_ONE : od_sc_b ? xa[0] : xa[l];
      end
      xc[l] <= cbuf[mi2_d.cb];
      xd[l] <= dbuf[mi2_d.cb];
    end
  end

  // the parts of the meta the later stages read: along the lane taps, and for the reductions
  typedef struct packed {
    logic             v;
    logic [2:0]       cls;       // f_cls(func)
    logic [LANES-1:0] mask;
    logic [AW-1:0]    waddr;
    logic             all_last;
  } mt_t;
  typedef struct packed {
    logic             v;
    logic [7:0]       tag;
    logic [AW-1:0]    waddr;
    logic             row_last;
    logic             all_last;
    logic             first;
    logic             final_;
    logic [7:0]       sub;
  } rm_t;
  rm_t m0r;
  assign m0r = '{v: m0.v, tag: m0.tag, waddr: m0.waddr, row_last: m0.row_last,
                 all_last: m0.all_last, first: m0.first, final_: m0.final_, sub: m0.sub};

  function automatic logic live(input rm_t m, input logic [7:0] t);
    return m.v && m.tag == t;
  endfunction

  // ------------------------------------------------------------------ elementwise lanes
  typedef struct packed {
    f32_t        v, k1, k2;
    logic [8:0]  ii;
    logic [2:0]  f;
  } lst_t;

  // Lane state fields live across slot s's delay line: read by a later slot or by the end tap
  // (EXP2 9: f v ii, RECIP 6: f k2 ii, RSQRT 10: f k2 ii, LOG2 10: f v) before being rewritten.
  // Derived from the slot programs in g_slot -- recheck when editing one. v is never live:
  // every slot followed by a read of v writes v itself. f is always live; dead fields are not
  // carried. LOG2 keeps t in k1 (slots 1..9), i2f(e) in k2 (slot 1's input stages to slot 9)
  // and e in ii (boundary 0 to slot 1's input stages). RECIP and RSQRT keep their result's
  // exponent adjustment in ii (RECIP's 127 - field, otpu_fp's rcp_ek; RSQRT's range scaling,
  // rsq_lo / rsq_hi).
  function automatic logic k1_live(input int s);
    return s <= 8;
  endfunction
  function automatic logic k2_live(input int s);
    return s <= 8;
  endfunction
  function automatic logic ii_live(input int s);
    return s <= 9;
  endfunction

  mt_t   mtap [NSX + 1];
  f32_t  lres [LANES];
  // MAX / MIN at tap 0 (short lanes): the lane's pick (lmv) enters the write data after every
  // other select (cw_data), which all come from flip-flops; lres leaves it out
  logic [LANES-1:0] lmm;                // the lane's result is lmv
  f32_t  lmv [LANES];

  // RSUM/RSSQ on the lanes' slot-0 multiply-add when the partial loop (RL cycles) holds the
  // input register, the multiplier and the adder plus a feedback flip-flop (DF of them);
  // otherwise (RL = 4) a squarer and an adder of their own per lane (see RSUM / RSSQ)
  localparam bit RMA = (RL >= 2 + LM + LA);
  localparam int DF  = RL - (1 + LM + LA);
  f32_t  pacc [LANES];                  // partial after adding this chunk's term
  f32_t  rsa [LANES], rsb [LANES];      // RMA: the term's factors, at m0
  f32_t  rfb [LANES];                   // RMA: the partial RL chunks ago (+0 for the first)

  assign mtap[0] = '{v: m0.v, cls: m0d.cls, mask: m0.mask, waddr: m0.waddr,
                     all_last: m0.all_last};
  // after slot 0 the lines carry the mask bits of lanes < NCL only (the others read 0): only
  // the composite functions get past tap 1, and they are issued on those lanes (iwid). Tap 1
  // keeps the whole mask.
  localparam int MTW = $bits(mt_t) - (LANES - NCL);
  function automatic logic [MTW-1:0] mt_pk(input mt_t m);
    return {m.v, m.cls, m.mask[NCL-1:0], m.waddr, m.all_last};
  endfunction
  function automatic mt_t mt_up(input logic [MTW-1:0] p);
    mt_t m;
    m = '0;
    {m.v, m.cls, m.mask[NCL-1:0], m.waddr, m.all_last} = p;
    return m;
  endfunction
  // slot 1 has three extra input stages for the EXP2 range reduction (clamp, floor, i2f)
  localparam int PRE1 = 3;
  // the lines' valids reset (otpu_vdelay: a power-up one would be a write at hit, below), the
  // rest (MTW - 1 bits, under v) in otpu_delay
  mt_t   msl [NSX];                          // meta at each slot's input mux
  for (genvar s = 0; s < NSX; s++) begin : g_mdel
    if (s == 1) begin : g_pre
      logic [MTW-1:0] p, q;
      assign p = mt_pk(mtap[s]);
      otpu_vdelay #(.N(PRE1)) u_pv (.clk, .rst, .en(en_b), .d(p[MTW-1]), .q(q[MTW-1]));
      otpu_delay #(.W(MTW - 1), .N(PRE1)) u_p (.clk, .en(en_b), .d(p[MTW-2:0]),
                                               .q(q[MTW-2:0]));
      assign msl[s] = mt_up(q);
    end else begin : g_nopre
      assign msl[s] = mtap[s];
    end
    if (s == 0) begin : g_d0
      localparam int MB = $bits(mt_t);
      logic [MB-1:0] p, q;
      assign p = msl[s];
      otpu_vdelay #(.N(SL)) u_dv (.clk, .rst, .en(en_b), .d(p[MB-1]), .q(q[MB-1]));
      otpu_delay #(.W(MB - 1), .N(SL)) u_d (.clk, .en(en_b), .d(p[MB-2:0]), .q(q[MB-2:0]));
      assign mtap[s + 1] = q;
    end else begin : g_dn
      logic [MTW-1:0] p, q;
      assign p = mt_pk(msl[s]);
      otpu_vdelay #(.N(SL)) u_dv (.clk, .rst, .en(en_b), .d(p[MTW-1]), .q(q[MTW-1]));
      otpu_delay #(.W(MTW - 1), .N(SL)) u_d (.clk, .en(en_b), .d(p[MTW-2:0]), .q(q[MTW-2:0]));
      assign mtap[s + 1] = mt_up(q);
    end
  end

  // the taps where functions end: 0 (MAX..FILL), T_EW (ADD/SUB/RSUB/MUL), T_RC, T_EX, T_RS
  localparam int T_EW = n_slots(V_ADD), T_RC = n_slots(V_RECIP), T_EX = n_slots(V_EXP2),
                 T_RS = n_slots(V_RSQRT);
  function automatic logic end_tap(input int s);
    return s == 0 || s == T_EW || s == T_RC || s == T_EX || s == T_RS;
  endfunction

  // which boundary holds an entry at its function's last slot (at most one: see header); each
  // end tap is tied to its functions, so the lanes read every result from a fixed tap. Past
  // tap 0 the class decides (the reductions' C_OTH has no slots).
  // (C8: a composite's chunk ends in otpu_se_comp, c_ov; the latency rule keeps it apart)
  logic hit [NSX + 1];
  mt_t  mo;
  always_comb begin
    mo = '0;
    for (int s = 0; s <= NSX; s++) begin
      if (s == 0) hit[s] = mtap[s].v && m0d.s0;
      else        hit[s] = end_tap(s) && mtap[s].v && n_slots(cls_f(mtap[s].cls)) == 4'(s);
      if (hit[s]) mo = mtap[s];
    end
    if (C8 && c_ov) mo = '{v: 1'b1, cls: C_OTH, mask: c_om, waddr: c_ometa[AW:1],
                           all_last: c_ometa[0]};
  end

  // long lanes: all functions
  for (genvar l = 0; l < NLL; l++) begin : g_lane
    localparam int NS = NSLOT;                     // slots of this lane
    f32_t x, y;
    lst_t st [NS + 1];
    assign x = xa[l];
    assign y = (m0.bmode == B_SCALAR) ? m0.imm : (m0.bmode == B_ROW) ? xb[0] : xb[l];

    // boundary 0: the simple functions' results and the composite functions' setup
    always_comb begin
      f32_t xz, ax;
      xz = ftz(x);
      ax = {1'b0, xz[30:0]};
      st[0] = '0;
      unique case (1'b1)
        m0d.fh[V_MAX], m0d.fh[V_MIN]: st[0].v = fp_mm(x, y, m0d.fh[V_MIN]);
        m0d.fh[V_COPY]:  st[0].v = xz;
        m0d.fh[V_ABS]:   st[0].v = fabs(x);
        m0d.fh[V_FILL]:  st[0].v = ftz(y);
        m0d.fh[V_RECIP]: begin                               // on the significand a
          st[0].k1 = {9'h17F, xz[22:0]};                     // -a
          st[0].k2 = RECIP_MAGIC - {9'h07F, xz[22:0]};       // seed
          st[0].ii = rcp_ek(xz);                             // y * 2^(127 - field)
          st[0].f  = {xz[31], (ax >= 32'h7E80_0000), (ax == 0)};
        end
        m0d.fh[V_RSQRT]: begin                               // on 16x / x / 16 (rsq_lo / hi)
          logic lo, hi;
          lo = rsq_lo(xz);
          hi = rsq_hi(xz);
          st[0].k2 = (lo ? RSQRT_MAGIC - 32'h0100_0000 : hi ? RSQRT_MAGIC + 32'h0100_0000
                         : RSQRT_MAGIC) - (xz >> 1);
          st[0].ii = lo ? 9'sd2 : hi ? -9'sd2 : 9'd0;         // y * 4, y / 4
          st[0].f  = {2'b00, (xz[31] || xz[30:0] == 0 || xz == F_INF)};
        end
        m0d.fh[V_LOG2]: begin                                // m in [sqrt(1/2), sqrt(2)), e
          logic ge, z;
          ge = (xz[22:0] >= LOG2_SQRT2);
          z  = (xz[30:0] == 0);
          st[0].k1 = {1'b0, ge ? 8'd126 : 8'd127, xz[22:0]};
          st[0].ii = 9'(xz[30:23]) - 9'd127 + 9'(ge);
          st[0].f  = {!z && (xz[31] || is_nan(xz)), xz == F_INF, z};   // NaN, +inf, -inf
        end
        default: ;
      endcase
    end

    for (genvar s = 0; s < NS; s++) begin : g_slot
      f32_t       ia, ib, ic_, ie, r;
      lst_t       sin_, sd, sti;
      logic [1:0] dest;                  // 0: none, 1: v, 2: k1, 3: k2
      logic       negd;                  // store fneg(result)
      if (s == 1) begin : g_pre
        // EXP2 range reduction: xf = x clamped, i = floor(xf) | -i2f(i) (into k2, unused here)
        // three stages: clamp | floor | i2f
        // (LOG2: i2f(e) into k2, from its ii)
        lst_t p0, p1, p2;
        logic e1, e2;                    // the entry in p0 / p1 is an EXP2
        logic g1, g2;                    // ... a LOG2
        always_ff @(posedge clk) if (en_l[l]) begin
          lst_t t;
          logic lo, hi;
          t = st[s];
          if (mtap[s].cls == C_EXP) begin
            lo = fp_gt(F_M126, t.v);
            hi = !fp_gt(F_128, t.v);
            t.v = (lo || hi) ? F_ZERO : t.v;
            t.f = {1'b0, hi, lo};
          end
          p0 <= t;
          e1 <= (mtap[s].cls == C_EXP);
          g1 <= (mtap[s].cls == C_LOG);
          t = p0;
          if (e1) t.ii = 9'(ffloor(p0.v));
          p1 <= t;
          e2 <= e1;
          g2 <= g1;
          t = p1;
          if (e2 || g2) begin
            t.k2 = i2f(32'($signed(p1.ii)));
            t.k2[31] = t.k2[31] ^ e2;              // EXP2: -i2f(i)
          end
          p2 <= t;
        end
        assign sti = p2;
      end else begin : g_nopre
        assign sti = st[s];
      end
      // the slot's function (one-hot): m0's at slot 0, the class's after it
      logic [NVF-1:0] sf;
      assign sf = (s == 0) ? m0d.fh : foh(cls_f(msl[s].cls));
      always_comb begin
        lst_t t;
        t = sti;
        ia = F_ZERO; ib = F_ONE; ic_ = F_NZ; ie = F_ONE; dest = 2'd0; negd = 1'b0;
        unique case (1'b1)
          sf[V_MUL]:  if (s == 0) begin ia = x; ib = y; dest = 2'd1; end
          sf[V_OUTER]: if (s == 0) begin                 // (OUTER1: B(r) * C(c) + dst)
            if (OUTER) begin ia = x; ib = xd[l]; ic_ = y; ie = xc[l]; end
            else begin ia = y; ib = xc[l]; ic_ = x; end
            dest = 2'd1;
          end
          sf[V_MAC]:  if (s == 0) begin                // A * B(c) + dst (port B: dst)
            ia = x; ib = (m0.bmode == B_SCALAR) ? m0.imm : xc[l]; ic_ = xb[l]; dest = 2'd1;
          end
          sf[V_ADD]:  if (s == 0) begin ia = x; ic_ = y; dest = 2'd1; end
          sf[V_SUB]:  if (s == 0) begin ia = x; ic_ = fneg(y); dest = 2'd1; end
          sf[V_RSUB]: if (s == 0) begin ia = y; ic_ = fneg(x); dest = 2'd1; end
          sf[V_RSUM], sf[V_RSSQ], sf[V_RDOT]:
            if (s == 0 && RMA) begin ia = rsa[l]; ib = rsb[l]; ic_ = rfb[l]; end
          sf[V_EXP2], sf[V_EXP2SUB]: begin
            if (s == 0) begin
              ia = x; ic_ = sf[V_EXP2SUB] ? fneg(y) : F_NZ; dest = 2'd1;
            end else if (s == 1) begin
              ia = t.v; ic_ = t.k2; dest = 2'd2;                          // f = xf - i
            end else if (s == 2) begin
              ia = EXP2_C7; ib = t.k1; ic_ = EXP2_C6; dest = 2'd1;
            end else if (s <= 8) begin
              ia = t.v; ib = t.k1; dest = 2'd1;
              case (s)
                3: ic_ = EXP2_C5;
                4: ic_ = EXP2_C4;
                5: ic_ = EXP2_C3;
                6: ic_ = EXP2_C2;
                7: ic_ = EXP2_C1;
                default: ic_ = EXP2_C0;
              endcase
            end
          end
          sf[V_RECIP]: if (s < 6) begin
            if (s % 2 == 0) begin                                       // 2 - ay; 1 - ay (s 4)
              ia = t.k1; ib = t.k2; ic_ = (s == 4) ? F_ONE : F_TWO; dest = 2'd1;
            end else begin                                              // y = y*t; y*e + y (s 5)
              ia = t.k2; ib = t.v; dest = 2'd3;
              if (s == 5) ic_ = t.k2;
            end
          end
          sf[V_RSQRT]: begin
            if (s == 0) begin                                                             // -h
              ia = rsq_lo(ftz(x)) ? 32'h4100_0000 : rsq_hi(ftz(x)) ? 32'h3D00_0000 : F_HALF;
              ib = ftz(x); dest = 2'd2; negd = 1'b1;           // (x 8, x / 32: h of the scaled x)
            end
            else if ((s - 1) % 3 == 0) begin ia = t.k2; ib = t.k2; dest = 2'd1; end       // y*y
            else if ((s - 1) % 3 == 1) begin                         // 1.5 - h yy; 0.5 - h yy (s 8)
              ia = t.k1; ib = t.v; ic_ = (s == 8) ? F_HALF : F_1P5; dest = 2'd1;
            end else begin                                              // y = y*c; y*e + y (s 9)
              ia = t.k2; ib = t.v; dest = 2'd3;
              if (s == 9) ic_ = t.k2;
            end
          end
          sf[V_LOG2]: begin
            if (s == 0) begin ia = t.k1; ic_ = F_M1; dest = 2'd2; end                 // t = m - 1
            else if (s == 1) begin ia = LOG2_C9; ib = t.k1; ic_ = LOG2_C8; dest = 2'd1; end
            else if (s <= 8) begin
              ia = t.v; ib = t.k1; dest = 2'd1;
              case (s)
                2: ic_ = LOG2_C7;
                3: ic_ = LOG2_C6;
                4: ic_ = LOG2_C5;
                5: ic_ = LOG2_C4;
                6: ic_ = LOG2_C3;
                7: ic_ = LOG2_C2;
                default: ic_ = LOG2_C1;
              endcase
            end else begin ia = t.v; ib = t.k1; ic_ = t.k2; dest = 2'd1; end    // q*t + i2f(e)
          end
          default: ;
        endcase
        sin_ = t;
      end
      f32_t       ra, rb, rc;
      lst_t       sr;
      logic [1:0] dr, dd;
      logic       nr, nd;
      always_ff @(posedge clk) if (en_l[l]) begin
        ra <= ia; rb <= ib; sr <= sin_; dr <= dest; nr <= negd;
      end
      // rc has a sync reset so it can't join u_c's two stages in a shift-register LUT: the
      // adder's c operand then leaves a flip-flop (clk->Q 0.3 ns) instead of an SRL (1.5 ns).
      // F_NZ is every slot's default c, so bits constant across a slot's cases stay constant.
      always_ff @(posedge clk)
        if (rst)     rc <= F_NZ;
        else if (en_l[l]) rc <= ic_;
      if (s == 0 && OUTER) begin : g_mma
        // slot 0: a*b + c*e (OUTER's second product; e = 1.0 otherwise)
        f32_t re;
        always_ff @(posedge clk) if (en_l[l]) re <= ie;
        otpu_fmma #(.LM(LM), .LA(LA)) u_ma (.clk, .en(en_l[l]), .a(ra), .b(rb), .c(rc), .e(re), .y(r));
      end else begin : g_ma
        otpu_fmadd #(.LM(LM), .LA(LA)) u_ma (.clk, .en(en_l[l]), .a(ra), .b(rb), .c(rc), .y(r));
      end
      if (s == 0 && RMA) begin : g_pacc
        assign pacc[l] = r;
      end
      // the live fields only (see k1_live); the dead ones read as 0 and are never used
      assign sd.v = '0;
      if (k1_live(s)) begin : g_k1
        otpu_delay #(.W(32), .N(LM + LA)) u_k1 (.clk, .en(en_l[l]), .d(sr.k1), .q(sd.k1));
      end else begin : g_nk1
        assign sd.k1 = '0;
      end
      if (k2_live(s)) begin : g_k2
        otpu_delay #(.W(32), .N(LM + LA)) u_k2 (.clk, .en(en_l[l]), .d(sr.k2), .q(sd.k2));
      end else begin : g_nk2
        assign sd.k2 = '0;
      end
      if (ii_live(s)) begin : g_ii
        otpu_delay #(.W(9), .N(LM + LA)) u_ii (.clk, .en(en_l[l]), .d(sr.ii), .q(sd.ii));
      end else begin : g_nii
        assign sd.ii = '0;
      end
      otpu_delay #(.W(3), .N(LM + LA)) u_f (.clk, .en(en_l[l]), .d(sr.f), .q(sd.f));
      otpu_delay #(.W(3), .N(LM + LA)) u_dst (.clk, .en(en_l[l]), .d({dr, nr}), .q({dd, nd}));
      always_comb begin
        f32_t rr;
        rr = nd ? fneg(r) : r;
        st[s + 1] = sd;
        case (dd)
          2'd1: st[s + 1].v = rr;
          2'd2: st[s + 1].k1 = rr;
          2'd3: st[s + 1].k2 = rr;
          default: ;
        endcase
      end
    end

    // the result of the entry that ends at a tap this cycle: each composite function's result
    // from its own tap (EXP2 T_EX, RECIP T_RC, RSQRT T_RS), the others' v from tap 0 or T_EW
    f32_t ex_r, rc_r, rs_r, lg_r;
    lst_t te, tc, ts;
    assign te = st[T_EX];
    assign tc = st[T_RC];
    assign ts = st[T_RS];
    assign ex_r = te.f[0] ? F_ZERO : te.f[1] ? F_INF : (te.v + {te.ii, 23'd0});
    // RECIP's and RSQRT's results scaled back (ii: their exponent adjustment)
    f32_t rc_y, rs_y;
    assign rc_y = tc.k2 + {tc.ii, 23'd0};
    assign rs_y = ts.k2 + {ts.ii, 23'd0};
    assign rc_r = tc.f[0] ? F_ZERO : tc.f[1] ? {tc.f[2], 31'd0} : (tc.f[2] ? fneg(rc_y) : rc_y);
    assign rs_r = ts.f[0] ? F_ZERO : rs_y;
    assign lg_r = ts.f[0] ? F_NINF : ts.f[2] ? F_NAN : ts.f[1] ? F_INF : ts.v;
    assign lres[l] = hit[T_EX] ? ex_r : hit[T_RC] ? rc_r :
                     hit[T_RS] ? ((mtap[T_RS].cls == C_LOG) ? lg_r : rs_r) :
                     hit[0] ? st[0].v : st[T_EW].v;
    assign lmm[l] = 1'b0;
    assign lmv[l] = '0;
  end

  // short lanes: slot 0 only. The composite functions are never issued here (imask), so their
  // setup, operand cases and result muxes are left out. C8: every lane, and slot 0 takes
  // otpu_se_comp's operands when it presents a chunk (c_s0).
  for (genvar l = NLL; l < LANES; l++) begin : g_slane
    f32_t x, y;
    f32_t st [2];                                  // v at boundaries 0 and 1
    assign x = xa[l];
    assign y = (m0.bmode == B_SCALAR) ? m0.imm : (m0.bmode == B_ROW) ? xb[0] : xb[l];

    always_comb begin
      st[0] = '0;
      unique case (1'b1)
        m0d.fh[V_COPY]:  st[0] = ftz(x);
        m0d.fh[V_ABS]:   st[0] = fabs(x);
        m0d.fh[V_FILL]:  st[0] = ftz(y);
        default: ;                                 // MAX, MIN: lmv
      endcase
    end
    f32_t ia, ib, ic_, ie, ra, rb, rc, re;
    always_comb begin
      ia = F_ZERO; ib = F_ONE; ic_ = F_NZ; ie = F_ONE;
      unique case (1'b1)
        m0d.fh[V_MUL]:  begin ia = x; ib = y; end
        m0d.fh[V_OUTER]: if (OUTER) begin ia = x; ib = xd[l]; ic_ = y; ie = xc[l]; end
                         else begin ia = y; ib = xc[l]; ic_ = x; end      // (OUTER1)
        m0d.fh[V_MAC]:  begin ia = x; ib = (m0.bmode == B_SCALAR) ? m0.imm : xc[l]; ic_ = xb[l]; end
        m0d.fh[V_ADD]:  begin ia = x; ic_ = y; end
        m0d.fh[V_SUB]:  begin ia = x; ic_ = fneg(y); end
        m0d.fh[V_RSUB]: begin ia = y; ic_ = fneg(x); end
        m0d.fh[V_RSUM], m0d.fh[V_RSSQ], m0d.fh[V_RDOT]:
          if (RMA) begin ia = rsa[l]; ib = rsb[l]; ic_ = rfb[l]; end
        default: ;
      endcase
    end
    always_ff @(posedge clk) if (en_l[l]) begin
      ra <= c_s0 ? c_sa[0][l] : ia; rb <= c_s0 ? c_sb[0][l] : ib; re <= c_s0 ? c_se[0][l] : ie;
    end
    // sync reset keeps rc out of u_c's SRL (see g_slot)
    always_ff @(posedge clk)
      if (rst)     rc <= F_NZ;
      else if (en_l[l]) rc <= c_s0 ? c_sc[0][l] : ic_;
    if (OUTER) begin : g_mma
      otpu_fmma #(.LM(LM), .LA(LA)) u_ma (.clk, .en(en_l[l]), .a(ra), .b(rb), .c(rc), .e(re), .y(st[1]));
    end else begin : g_ma                       // e is 1.0 without OUTER: a*b + c
      otpu_fmadd #(.LM(LM), .LA(LA)) u_ma (.clk, .en(en_l[l]), .a(ra), .b(rb), .c(rc), .y(st[1]));
    end
    if (RMA) begin : g_pacc
      assign pacc[l] = st[1];
    end
    assign c_s0y[l] = st[1];

    // the result of the entry that ends at tap 0 or 1 this cycle (other taps: masked here), or
    // (C8) a composite's
    assign lres[l] = (C8 && c_ov) ? c_od[l] : hit[0] ? st[0] : st[1];
    assign lmm[l] = !(C8 && c_ov) && hit[0] && (m0d.fh[V_MAX] || m0d.fh[V_MIN]);
    assign lmv[l] = fp_mm(x, y, m0d.fh[V_MIN]);
    assign c_ia[l] = x;
    assign c_ib[l] = y;
  end

  if (C8) begin : g_comp
    f32_t sy [CNS][LANES];
    assign sy[0] = c_s0y;
    assign sy[1] = c_uy;
    assign sy[2] = c_qy;
    for (genvar k = 3; k < CNS; k++) begin : g_sy
      for (genvar l = 0; l < LANES; l++) begin : g_l
        assign sy[k][l] = '0;                  // (otpu_se_comp's own units: NX 3)
      end
    end
    logic [AW+FMW:0] imeta;
    if (FSUM) begin : g_fsx
      assign imeta = {m0.fs, m0.first, m0.row_last, m0.par, m0.sub[2:0], m0.waddr, m0.all_last};
    end else begin : g_nfsx
      assign imeta = {m0.waddr, m0.all_last};
    end
    otpu_se_comp #(.LANES(LANES), .MW(AW + 1 + FMW), .NS(CNS), .HA(3), .EXT(1'b1), .NX(3),
                   .SIG(VSIG)) u_comp (
      .clk, .rst, .en(en_b), .en_l, .in_v(m0.v && m0d.cf), .in_c(m0d.cc), .in_a(c_ia),
      .in_b(c_ib), .in_m(m0.mask), .in_meta(imeta), .hold(c_hold),
      .hold_n(c_hold_n),
      .st_sel(c_sel), .st_a(c_sa), .st_b(c_sb), .st_c(c_sc), .st_e(c_se), .st_y(sy),
      .out_v(c_ov), .out_d(c_od), .out_m(c_om), .out_meta(c_ometa));
    assign c_s0 = c_sel[0];
  end else begin : g_nocomp
    assign c_hold = 1'b0;
    assign c_hold_n = 1'b0;
    assign c_ov = 1'b0;
    assign c_s0 = 1'b0;
    assign c_sel = '0;
    assign c_om = '0;
    assign c_ometa = '0;
    for (genvar l = 0; l < LANES; l++) begin : g_z
      assign c_od[l] = '0;
      for (genvar k = 0; k < CNS; k++) begin : g_k
        assign c_sa[k][l] = '0;
        assign c_sb[k][l] = '0;
        assign c_sc[k][l] = '0;
        assign c_se[k][l] = '0;
      end
    end
  end

  // ------------------------------------------------------------------ RMAX
  // pairwise max tree over the lanes of the chunk (masked lanes drop out), one registered
  // stage per level: stage 0 from the TMEM read data (LANES -> HL), stage j halves HL >> (j-1)
  // Every level pairs neighbours (2l, 2l + 1), so the left element always covers the lower
  // lanes and keeping it on a tie keeps the lowest lane (pairing l with l + HL/2^j does not:
  // after level 0 element l may hold lane l + HL, above its partner's).
  localparam int HL = LANES / 2;
  localparam int ML = $clog2(HL);             // levels after the first
  // (ARGMAX: each value's lane, mxh_i, ties to the lower lane; the chunk's column along, mxh_c)
  f32_t  mxh_v [ML + 1][HL];                  // stage j holds HL >> j values
  logic  mxh_h [ML + 1][HL];
  logic [LW-1:0] mxh_i [ML + 1][HL];
  rm_t   mxh_m [ML + 1];
  logic [15:0] mxh_c [ML + 1];
  f32_t  mxc_q;
  rm_t   mxm_q;
  f32_t  mx_run;
  logic  mx_have;
  logic [16:0] mx_ri;                         // the running maximum's column
  always_ff @(posedge clk) if (en_b) begin
    f32_t v [LANES];
    logic h [LANES];
    logic up;
    for (int l = 0; l < LANES; l++) begin
      v[l] = ftz(xa[l]);
      h[l] = m0.mask[l];
    end
    for (int l = 0; l < HL; l++) begin
      up = !h[2*l] || (h[2*l + 1] && fp_gt(v[2*l + 1], v[2*l]));
      mxh_v[0][l] <= up ? v[2*l + 1] : v[2*l];
      mxh_h[0][l] <= h[2*l] || h[2*l + 1];
      mxh_i[0][l] <= up ? LW'(2*l + 1) : LW'(2*l);
    end
    mxh_c[0] <= m0.col;
    for (int j = 1; j <= ML; j++) begin
      for (int l = 0; l < (HL >> j); l++) begin
        up = !mxh_h[j-1][2*l] || (mxh_h[j-1][2*l + 1] &&
                                  fp_gt(mxh_v[j-1][2*l + 1], mxh_v[j-1][2*l]));
        mxh_v[j][l] <= up ? mxh_v[j-1][2*l + 1] : mxh_v[j-1][2*l];
        mxh_h[j][l] <= mxh_h[j-1][2*l] || mxh_h[j-1][2*l + 1];
        mxh_i[j][l] <= up ? mxh_i[j-1][2*l + 1] : mxh_i[j-1][2*l];
      end
      mxh_c[j] <= mxh_c[j-1];
    end
  end
  // the chunk's meta along the tree, its valid reset (a power-up one would be a live row's max);
  // the rest is data, read only through live() (its v and tag)
  always_ff @(posedge clk) begin
    if (en_b) begin
      mxh_m[0] <= m0r;
      for (int j = 1; j <= ML; j++) mxh_m[j] <= mxh_m[j-1];
    end
    if (rst) for (int j = 0; j <= ML; j++) mxh_m[j].v <= 1'b0;
  end
  assign mxc_q = mxh_v[ML][0];
  assign mxm_q = mxh_m[ML];
  // the running maximum keeps the older value on ties (the same bits for RMAX: in the total
  // order only equal bits tie; ARGMAX: the first column)
  wire  mx_keep = mx_have && !fp_gt(mxc_q, mx_run);
  f32_t mx_new;
  assign mx_new = mx_keep ? mx_run : mxc_q;
  wire [16:0] mx_ni = mx_keep ? mx_ri : 17'(mxh_c[ML]) + 17'(mxh_i[ML][0]);
  // ARGMAX's pair: A (the row's max and column), B (+ imm), C (sign, magnitude, leading
  // zeros), D (the fp32 index), then written; every stage advances on en
  logic        am1_v, am2_v, am3_v, am4_v, am1_l, am2_l, am3_l, am4_l;
  logic [AW-1:0] am1_a, am2_a, am3_a, am4_a;
  f32_t        am1_m, am2_m, am3_m, am4_m, am4_i;
  logic [16:0] am1_i;
  logic [31:0] am2_s, am3_x;
  logic        am3_s;
  logic [4:0]  am3_z;
  always_ff @(posedge clk)
    if (rst) begin
      am1_v <= 1'b0; am2_v <= 1'b0; am3_v <= 1'b0; am4_v <= 1'b0;
    end else if (en_b) begin
      am1_v <= red_act_b && is_am_b && live(mxm_q, tag_b) && mxm_q.row_last;
      am1_l <= mxm_q.all_last; am1_a <= mxm_q.waddr; am1_m <= mx_new; am1_i <= mx_ni;
      am2_v <= am1_v; am2_l <= am1_l; am2_a <= am1_a; am2_m <= am1_m;
      am2_s <= 32'(am1_i) + imm_b;
      am3_v <= am2_v; am3_l <= am2_l; am3_a <= am2_a; am3_m <= am2_m;
      begin
        logic [31:0] x;
        logic [4:0]  z;
        x = am2_s[31] ? -am2_s : am2_s;
        z = '0;
        for (int k = 0; k < 32; k++) if (x[k]) z = 5'(31 - k);
        am3_s <= am2_s[31]; am3_x <= x; am3_z <= z;
      end
      am4_v <= am3_v; am4_l <= am3_l; am4_a <= am3_a; am4_m <= am3_m;
      begin
        logic [31:0] n;
        logic [23:0] m;
        logic [7:0]  e;
        n = am3_x << am3_z;                   // the leading one at bit 31
        m = n[31:8];
        e = 8'd158 - 8'(am3_z);
        if (n[7] && ((|n[6:0]) || m[0])) begin
          if (m == 24'hFF_FFFF) begin m = 24'h80_0000; e = e + 8'd1; end
          else m = m + 24'd1;
        end
        am4_i <= (am3_x == 0) ? 32'd0 : {am3_s, e, m[22:0]};
      end
    end

  // ------------------------------------------------------------------ RSUM / RSSQ
  // RMA: the term enters the lane's slot 0 through its input register (ra/rb/rc), so it reaches
  // the adder one cycle later than a squarer of its own would
  // (v in otpu_vdelay, the rest in otpu_delay, as the lane taps')
  localparam int RB = $bits(rm_t);
  rm_t   mtq, mt;                       // meta at the adder inputs / aligned with `pacc`
  logic [RB-1:0] m0p, mtqp, mtp;
  assign m0p = m0r;
  otpu_vdelay #(.N(RMA ? LM + 1 : LM)) u_mtqv (.clk, .rst, .en(en_b), .d(m0p[RB-1]),
                                              .q(mtqp[RB-1]));
  otpu_delay #(.W(RB - 1), .N(RMA ? LM + 1 : LM)) u_mtq (.clk, .en(en_b), .d(m0p[RB-2:0]),
                                                         .q(mtqp[RB-2:0]));
  assign mtq = mtqp;
  otpu_vdelay #(.N(LA)) u_mtv (.clk, .rst, .en(en_b), .d(mtqp[RB-1]), .q(mtp[RB-1]));
  otpu_delay #(.W(RB - 1), .N(LA)) u_mt (.clk, .en(en_b), .d(mtqp[RB-2:0]), .q(mtp[RB-2:0]));
  assign mt = mtp;
  logic  first_e;                       // !RMA: mtq.first one cycle early
  if (RMA) begin : g_nfe
    assign first_e = 1'b0;
  end else begin : g_fe
    otpu_delay #(.W(1), .N(LM - 1)) u_first (.clk, .en(en_b), .d(m0.first), .q(first_e));
  end
  for (genvar l = 0; l < LANES; l++) begin : g_red
    f32_t sa, sb, yb;
    // sign/exponent masked; the significand raw from xa (the multiplier flushes a zero exponent
    // itself and reads the significand only when both exponents are normal), so the term is
    // the same as that of m0.mask[l] ? ftz(xa[l]) : +0. RDOT's B is 1.0 on masked lanes (a
    // padding term 0 * B must stay +0 when B is infinite).
    assign sa = {m0.mask[l] & xa[l][31], m0.mask[l] ? xa[l][30:23] : 8'd0, xa[l][22:0]};
    assign yb = (m0.bmode == B_SCALAR) ? m0.imm : (m0.bmode == B_ROW) ? xb[0] : xb[l];
    assign sb = m0.sq ? sa : (m0.dot && m0.mask[l]) ? yb : F_ONE;
    // stream mode: the term is the tail's S * a (bubbles come only after the last row, and
    // their partials are never captured)
    assign rsa[l] = (SE && sa_r[l]) ? s_xd[l] : sa;
    assign rsb[l] = (SE && sa_r[l]) ? s_xa[l] : sb;
    // pacc(chunk c) = pacc(chunk c - RL) + term(c): a loop of exactly RL cycles
    if (RMA) begin : g_ma
      // through slot 0's u_ma (term + prev: the add is commutative bit for bit), fed back by
      // fbq, which slot 0 reads (as c) the cycle after it's written: cleared on the chunks that
      // are on m0 then (mi2), the first ones' +0. rc keeps the adder's c off a shift register.
      // A stream clears c instead, on its X segment's first flag (mi2 is idle then).
      f32_t fbq, fbd;
      otpu_delay #(.W(32), .N(DF - 1)) u_fb (.clk, .en(en_l[l]), .d(pacc[l]), .q(fbd));
      always_ff @(posedge clk) if (en_l[l]) fbq <= mi2_d.first ? F_ZERO : fbd;
      assign rfb[l] = s_first ? F_ZERO : fbq;
    end else begin : g_acc
      f32_t tq, prev;
      assign rfb[l] = F_ZERO;
      otpu_fmul #(.LAT(LM)) u_sq (.clk, .en(en_l[l]), .a(sa), .b(sb), .y(tq));
      if (RL - LA >= 1) begin : g_fbq
        // the loop's last stage is a flip-flop with a sync clear (the first chunks' +0): no
        // shift-register LUT and no mux in front of the adder
        f32_t fbq, fbd;
        otpu_delay #(.W(32), .N(RL - LA - 1)) u_fb (.clk, .en(en_l[l]), .d(pacc[l]), .q(fbd));
        always_ff @(posedge clk) if (en_l[l]) fbq <= first_e ? F_ZERO : fbd;
        assign prev = fbq;
      end else begin : g_fbw
        f32_t fb;
        otpu_delay #(.W(32), .N(RL - LA)) u_fb (.clk, .en(en_l[l]), .d(pacc[l]), .q(fb));
        assign prev = mtq.first ? F_ZERO : fb;
      end
      otpu_fadd #(.LAT(LA)) u_acc (.clk, .en(en_l[l]), .a(prev), .b(tq), .y(pacc[l]));
    end
  end

  // ------------------------------------------------------------------ FSUM
  // An EXP2SUB's row sums (see the header). Each lane keeps the partials of two rows in
  // distributed RAM (bank: the row's parity; word: the chunk's index in its row mod 8, partial
  // sub * 8 + l). A result chunk off otpu_se_comp reads its partial (+0 on the row's first 8
  // chunks: +0 + x, the reference's first term), adds its result on the lane's own adder and
  // writes it back LA + 1 cycles on (masked lanes keep theirs: their +0 terms change nothing,
  // EXP2 results are never -0 or NaN); a partial's next term is 8 chunks later. A row's final
  // partials (its last chunk written) drain into u_vt, sub 0..7 on 8 cycles, from phase 0 of
  // an 8-cycle counter (rows a multiple of RL apart). A row so drains within 15 cycles of its
  // last write, each word before the row after the next one (8 chunks or more later, cols >=
  // 64) writes it; its root goes to the row-sum buffer.
  logic       fs_cap, fs_rl;                 // a drained sub on u_vt's input (fs_pd)
  logic [2:0] fs_sub;
  f32_t       fs_pd [LANES];
  logic       begf_b;                        // an FSUM began (the back end's, as begr_b)
  logic       ofs;                           // the write mux's chunk (c_ov) is summed
  if (FSUM && C8) begin : g_fsum
    typedef struct packed { logic rl, p; logic [2:0] s; logic [LANES-1:0] m; } fm_t;
    wire       in_f = c_ometa[AW+6];
    fm_t       s1, s5;                        // a chunk at the adders' inputs / outputs
    logic      s1v, s5v;
    logic [LA-1:0] vsr;
    logic [2:0] ph, ds;                       // the phase; the sub being drained
    logic [1:0] pend;                         // a row's partials are final (by parity)
    logic       dpar, d_on;                   // the next row to drain; draining
    wire        d_go = !d_on && pend[dpar] && ph == 3'd0;
    wire  [2:0] d_sub = d_on ? ds : 3'd0;
    assign ofs = c_ov && c_ometa[AW+7];
    always_ff @(posedge clk) if (en_b) s1 <= '{rl: c_ometa[AW+5], p: c_ometa[AW+4],
                                               s: c_ometa[AW+3:AW+1], m: c_om};
    otpu_delay #(.W($bits(fm_t)), .N(LA)) u_s5 (.clk, .en(en_b), .d(s1), .q(s5));
    assign s5v = vsr[LA-1];
    always_ff @(posedge clk)
      if (rst) begin
        s1v <= 1'b0; vsr <= '0;
        ph <= '0; pend <= '0; dpar <= 1'b0; d_on <= 1'b0; fs_cap <= 1'b0;
      end else begin
        if (en_b) begin
          s1v <= ofs;
          vsr <= {vsr[LA-2:0], s1v};
          ph <= ph + 3'd1;
          fs_cap <= d_go || d_on;
          fs_sub <= d_sub;
          fs_rl <= (d_sub == 3'd7);
          if (d_go) begin
            d_on <= 1'b1; ds <= 3'd1; pend[dpar] <= 1'b0;
          end else if (d_on) begin
            ds <= ds + 3'd1;
            if (ds == 3'd7) begin d_on <= 1'b0; dpar <= ~dpar; end
          end
          if (s5v && s5.rl) pend[s5.p] <= 1'b1;
        end
        if (begf_b) dpar <= 1'b0;             // (row 0's parity; nothing is pending then)
      end
    for (genvar l = 0; l < LANES; l++) begin : g_l
      (* ram_style = "distributed" *) f32_t acc [16];
      f32_t oa, ob, y;
      always_ff @(posedge clk) if (en_l[l]) begin
        oa <= in_f ? F_ZERO : acc[c_ometa[AW+4:AW+1]];
        ob <= c_od[l];
        fs_pd[l] <= acc[{dpar, d_sub}];
      end
      otpu_fadd #(.LAT(LA)) u_add (.clk, .en(en_l[l]), .a(oa), .b(ob), .y(y));
      always_ff @(posedge clk) if (en_l[l] && s5v && s5.m[l]) acc[{s5.p, s5.s}] <= y;
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk)
      if (!rst && en_b) begin
        if (ofs && cwid != 16'(LANES))
          $fatal(1, "otpu_vpu: FSUM needs composite chunks of LANES columns (+vt_cw)");
        if (s5v && s5.rl && pend[s5.p] && !(d_go && dpar == s5.p))
          $fatal(1, "otpu_vpu: FSUM: a row's partials are final before the last drained");
        if (s5v && (d_go || d_on) && s5.p == dpar && s5.s > d_sub)
          $fatal(1, "otpu_vpu: FSUM: a partial is written before it drained (cols < 64?)");
      end
`endif
  end else begin : g_nfsum
    assign ofs = 1'b0;
    assign fs_cap = 1'b0;
    assign fs_rl = 1'b0;
    assign fs_sub = '0;
    for (genvar l = 0; l < LANES; l++) begin : g_z
      assign fs_pd[l] = '0;
    end
  end

  // the folding tree (otpu_vtree): a row's sum RD cycles after its last final partial
  // a final partial of this reduction (or of the stream's dot A) is on `pacc`, or one of the
  // tail's dot Q, in a window of its own (q_cap)
  wire  a_cap = (SE && ss_act) ? s_mt.v && s_mt.final_ :
                                 red_act_b && is_sum_b && live(mt, tag_b) && mt.final_;
  // (FSUM: a drained row's partials, fs_cap; nothing else uses u_vt then)
  wire  cap = a_cap || q_cap || fs_cap;
  wire  vt_rl = fs_cap ? fs_rl : q_cap ? q_rl : (SE && ss_act) ? s_mt.row_last : mt.row_last;
  wire  [7:0] vt_sub = fs_cap ? 8'(fs_sub) : q_cap ? q_sub : (SE && ss_act) ? s_mt.sub : mt.sub;
  f32_t vt_in [LANES];
  for (genvar l = 0; l < LANES; l++) begin : g_vtin
    assign vt_in[l] = fs_cap ? fs_pd[l] : q_cap ? q_pd[l] : pacc[l];
  end
  // root_o: the root from a copy of its register, for the DMA's O path alone (the tail's Q dot,
  // qo_d -> o_d -> ss_o_d): with no load here it can sit between the VPU and the DMA, so the
  // wire to the DMA's o_q spans two cycles (133.33 MHz, 110ec6d: u_vt root -> u_dma o_q, 0
  // levels, 6.6 ns of route, +0.102 ns)
  f32_t root, root_o;
  logic root_v;
  otpu_vtree #(.LANES(LANES), .LA(LA)) u_vt (.clk, .rst(rst || s_init), .en(en_b), .pacc(vt_in), .cap,
                                              .row_last(vt_rl), .sub(vt_sub), .root, .root_o,
                                              .root_v);
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && rst_seen && en_b && a_cap && q_cap) $fatal(1, "otpu_vpu: A and Q windows meet on u_vt");
  bit rt_ld;                                  // (root and its copy load together: after a load)
  initial rt_ld = 1'b0;
  always @(posedge clk) if (en_b) rt_ld <= 1'b1;
  always_ff @(posedge clk)
    if (!rst && rst_seen && en_b && fs_cap && (a_cap || q_cap))
      $fatal(1, "otpu_vpu: FSUM meets a sum on u_vt");
  always_ff @(posedge clk)
    if (rt_ld && root_o != root) $fatal(1, "otpu_vpu: root_o %h is not root %h", root_o, root);
`endif

  // rows finish in order: the next root goes to wr_row
  logic [AW-1:0] wr_row;
  logic [15:0]   rows_done;
  // RDOT's row-sum buffer: row i in bank i % LANES; the flush writes rows fl .. fl+LANES-1
  // (FSUM's sums too: at imm, the EXP2SUB's w7 + R[rd])
  localparam int NRB = 256;
  logic          rdb;                        // this reduction's row sums are buffered
  logic          flushing;
  logic [15:0]   fl;
  // a flush write this cycle (FSUM's waits for a cycle with no result to write: the VOPs after
  // the EXP2SUB run during it), and the rows flushed
  wire           fl_w = flushing && !(fs_b && mo.v);
  wire  [15:0]   fl_rows = fs_b ? fs_rows : rows_b;
  f32_t          rbuf_q [LANES];
  for (genvar l = 0; l < LANES; l++) begin : g_rbuf
    // distributed RAM: the flush reads a row sum the cycle after its write (the last row's).
    // With the write buffer (WBUF) behind the read, Vivado made this a block RAM with fl as its
    // read-address register, which returns the old word on that collision: every RDOT of at
    // most LANES rows then wrote its last row sum stale (on the card, not in simulation)
    (* ram_style = "distributed" *) f32_t rb [NRB / LANES];
    always_ff @(posedge clk)
      if (en_b && ((red_act_b && is_sum_b && rdb) || fs_b) && root_v &&
          rows_done[LW-1:0] == LW'(l))
        rb[rows_done[15:LW]] <= root;
    assign rbuf_q[l] = rb[fl[15:LW]];
  end

  // ------------------------------------------------------------------ TMEM writes
  // The writes are computed here (cw_*) and registered (tw_*): a TMEM write is performed one
  // granted cycle after the cycle that produced it, so no path runs from the TMEM read data
  // through the lanes into the TMEM write port. `done` follows its instruction's last write.
  logic [LANES-1:0]         cw_en;
  logic [LANES-1:0][AW-1:0] cw_addr, tw_a;
  logic [LANES-1:0][31:0]   cw_data;
  logic                     done_i, dpend;
  logic                     empty_b;       // an empty instruction began (FB: a cycle late)
  for (genvar l = 0; l < LANES; l++) begin : g_twa
    assign tw_addr[l] = 32'(tw_a[l]);
  end
  logic                     wb_e;          // no write pending
  if (!WBUF) begin : g_wdir
    assign en = (SE && ss_act) ? sen : gnt && f_ok;
    assign en_l = {LANES{en}};
    assign en_b = en;
    assign wb_e = (tw_en == '0);
    always_ff @(posedge clk) begin
      if (rst) begin
        tw_en <= '0;
        done <= 1'b0; dpend <= 1'b0;
      end else begin
        if (gnt) begin
          tw_en <= f_ok ? cw_en : '0; tw_a <= cw_addr; tw_data <= cw_data;
        end
        done <= (done_i || dpend) && gnt;
        dpend <= (done_i || dpend) && !gnt;
      end
    end
  end else begin : g_wbuf
    // Two slots, written at the tail and presented at the head (pointers, no data moves: the
    // grant only reaches the pointers and the count). An enabled cycle pushes its writes (if
    // any) with the pending done; `done` follows the pop of the entry that carries it. The
    // enable for the next cycle is room for a push even if the head is not taken then.
    // SE: the enable is computed a cycle ahead (en_q) and en_r is its copy, so en_r's replicas
    // have a flip-flop in front of them, no logic; it is then a cycle staler, and the buffer
    // has four slots (at most three are used). In stream mode it is the stream's pe (two
    // cycles behind the DMA's, see the header).
    // FB: the back end pushes a cycle after the front's enable, so the room is decided a cycle
    // further ahead: a push at t + 2 (SE: + 3) on a count < 2 at t, up to four (three) in all
    localparam int NWQ = FB ? (SE ? 8 : 4) : (SE ? 4 : 2);
    localparam int WQW = $clog2(NWQ);
    typedef struct packed {
      logic [LANES-1:0]         en;
      logic [LANES-1:0][AW-1:0] a;
      logic [LANES-1:0][31:0]   d;
      logic                     dn;
    } wb_t;
    wb_t        wq [NWQ];
    logic [WQW-1:0] wh, wt;                      // head, tail slot
    logic [WQW:0]   wn;                          // entries
    (* max_fanout = 64 *) logic en_r;
    (* max_fanout = 32 *) logic en_q;
    assign en = en_r;
    assign wb_e = (wn == 0);
    wire  pdn  = done_i || dpend;
    wire  push = en_b && ((|cw_en) || pdn);
    wire  pop  = (wn != 0) && gnt;
    // the count without a pop and with one, so the grant (pop) only picks: it enters the count
    // and the enables at their last LUT
    wire  [WQW:0] wn_p0 = wn + (WQW+1)'(push);
    wire  [WQW:0] wn_p1 = wn_p0 - 1'b1;
    wire  [WQW:0] wn_nx = pop ? wn_p1 : wn_p0;
    wb_t  hd, nh;                                // the head, the entry after it
    assign hd = wq[wh];
    assign nh = wq[wh + 1'b1];
    // the head's lane enables (none when empty) and lane 0's bank in flip-flops, loaded as the
    // head changes: the TMEM port's rotation and the arbiter's bank mask start at them instead
    // of at the LUT RAM's read. A run's lanes are contiguous, so lane l's bank is hrot + l.
    logic [LANES-1:0] hen;
    logic [LW-1:0]    hrot;
    always_comb begin
      tw_en = hen;
      tw_a = hd.a;
      for (int l = 0; l < LANES; l++) tw_a[l][LW-1:0] = hrot + LW'(l);
      tw_data = hd.d;
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && rst_seen) begin
        if (hen != ((wn != 0) ? hd.en : '0)) $fatal(1, "otpu_vpu: hen is not the head's enables");
        for (int l = 0; l < LANES; l++)
          if (hen[l] && hd.a[l][LW-1:0] != hrot + LW'(l))
            $fatal(1, "otpu_vpu: WBUF lane %0d's bank is not hrot + %0d", l, l);
      end
`endif
    always_ff @(posedge clk) if (push) wq[wt] <= '{en: cw_en, a: cw_addr, d: cw_data, dn: pdn};
    // en_r's next value, and its copies per lane (kept instances: identical flip-flops merge),
    // loaded with it: en_r's max_fanout replicas each reached 64 registers anywhere in the VPU
    // (r22's m8: into the tail's units, 7.0 ns of route with no logic; at 150 MHz, 977 endpoints)
    wire en_rd = SE ? en_q && !rhold : (pop ? (wn_p1 < 2) : (wn_p0 < 2)) && red_thr && f_ok &&
                                       !rhold;
    // the back end's next enable (FB): en, a cycle late (en_r); in stream mode the stream's,
    // which the front's en is then too (en_q: rhold is low, the front reads nothing) -- either
    // way a flip-flop, so its copies have no logic in front of them. Else en's (en_rd).
    wire en_bd = !FB ? en_rd : (SE && ss_act) ? en_q : en_r;
    for (genvar l = 0; l < LANES; l++) begin : g_enl
      (* keep_hierarchy = "yes" *)
      otpu_delay #(.W(1), .N(1)) u_en (.clk, .en(1'b1), .d(!rst && en_bd), .q(en_l[l]));
    end
    if (FB) begin : g_enb
      (* max_fanout = 64 *) logic en_bq;
      always_ff @(posedge clk) en_bq <= !rst && en_bd;
      assign en_b = en_bq;
    end else begin : g_nenb
      assign en_b = en_r;
    end
`ifndef SYNTHESIS
    always @(posedge clk) if (rst_seen && en_l != {LANES{en_b}}) $fatal(1, "otpu_vpu: en_l is not en_b");
`endif
    always_ff @(posedge clk) begin
      if (rst) begin
        wh <= '0; wt <= '0; wn <= '0; en_r <= 1'b0; en_q <= 1'b0;
        done <= 1'b0; dpend <= 1'b0; hen <= '0;
      end else begin
        if (push) wt <= wt + 1'b1;
        if (pop) wh <= wh + 1'b1;
        if (push && (wn == 0 || (wn == 1 && pop))) begin      // the push is the new head
          hen <= cw_en;
          hrot <= cw_addr[0][LW-1:0];
        end else if (pop) begin                                // the next entry, or none
          hen <= (wn > 1) ? nh.en : '0;
          hrot <= nh.a[0][LW-1:0];
        end
        wn <= wn_nx;
        if (SE) begin
          en_q <= ss_act_d ? ss_pe : (pop ? (wn_p1 < 2) : (wn_p0 < 2)) && red_thr && f_ok;  // a stream: its pe (as sen)
          en_r <= en_rd;
        end else begin
          en_r <= en_rd;
        end
        done <= pop && hd.dn;
        dpend <= pdn && !en_b;
      end
    end
  end
  always_comb begin
    logic [LANES-1:0] mm;                        // the lane writes lmv
    cw_en = '0; cw_addr = '0; cw_data = '0; mm = '0;
    if (mo.v) begin
      for (int l = 0; l < LANES; l++) begin
        if (mo.mask[l]) begin
          cw_en[l] = 1'b1;
          cw_addr[l] = mo.waddr + AW'(l);
          cw_data[l] = lres[l];
          mm[l] = lmm[l];
        end
      end
    end
    if (red_act_b && func_b == V_RMAX && live(mxm_q, tag_b) && mxm_q.row_last) begin
      cw_en[0] = 1'b1;
      cw_addr[0] = mxm_q.waddr;
      cw_data[0] = mx_new;
      mm[0] = 1'b0;
    end
    if (am4_v) begin                             // ARGMAX: the row's (max, index) pair
      cw_en[1:0] = 2'b11;
      cw_addr[0] = am4_a;
      cw_addr[1] = am4_a + AW'(1);
      cw_data[0] = am4_m;
      cw_data[1] = am4_i;
      mm[1:0] = 2'b00;
    end
    if (red_act_b && is_sum_b && root_v && !rdb) begin
      cw_en[0] = 1'b1;
      cw_addr[0] = wr_row;
      cw_data[0] = root;
      mm[0] = 1'b0;
    end
    if (fl_w) begin
      for (int l = 0; l < LANES; l++) begin
        if (32'(fl) + 32'(l) < 32'(fl_rows)) begin
          cw_en[l] = 1'b1;
          cw_addr[l] = (fs_b ? AW'(fs_imm) : dst_b) + AW'(fl) + AW'(l);
          cw_data[l] = rbuf_q[l];
          mm[l] = 1'b0;
        end
      end
    end
    for (int l = 0; l < LANES; l++) if (mm[l]) cw_data[l] = lmv[l];
  end

  // ------------------------------------------------------------------ stream engine
  // The grant: a stream asks (ss_req; rdy falls a cycle later, with ss_rq), and once nothing is
  // starting (the sequencer's start follows rdy a cycle late), queued, issuing, reducing, in the
  // lanes or waiting to be written (nor a done pending), ss_act rises (s_init with it) and holds
  // until ss_req falls (ss_gnt falls the next cycle). The DMA fills and streams only after
  // ss_gnt, and drops ss_req after its last O, when the stream's partials, kv and tail have
  // drained. An empty instruction (FSUM's take) leaves the queue at its begin and completes at
  // empty_b, which FB registers: the cycle between has no done yet, so it counts as busy.
  if (SE) begin : g_se
    wire idle = !start && (cq_n == 0) && !busy && !empty_b && !done_i && !dpend && wb_e;
    assign ss_act_d = SE_SS && ss_req && ss_rq && (ss_act || idle);
    // the inputs through two registers (sen_q, s1_*: the first), as the enable (en_q, en_r)
    (* max_fanout = 32 *) logic sen_q;
    logic       s1_in_v;
    f32_t       s1_in_d [LANES], s1_fd [LANES];
    logic [2:0] s1_fk;
    logic [4:0] s1_fi;
    always_ff @(posedge clk) begin
      if (rst) begin
        ss_rq <= 1'b0; ss_act <= 1'b0; s_init <= 1'b0; sen_q <= 1'b0; sen <= 1'b0;
        s1_fk <= SF_NONE; s_fk <= SF_NONE;
      end else begin
        ss_rq <= SE_SS && ss_req;
        ss_act <= ss_act_d;
        s_init <= SE_SS && ss_req && ss_rq && !ss_act && idle;
        sen_q <= ss_act_d && ss_pe;
        sen <= sen_q;
        s1_fk <= ss_act ? ss_fk : SF_NONE;
        s_fk <= s1_fk;
      end
      s1_fi <= ss_fi; s1_fd <= ss_fd; s_fd <= s1_fd;
      s1_in_v <= ss_in_v; s1_in_d <= ss_in_d; s_in_v <= s1_in_v; s_in_d <= s1_in_d;
      s_cfg <= ss_cfg;
    end
    assign ss_gnt = ss_act;
    // ss_act's copies for the lanes: ss_act selects every lane's RDOT term (64 selects a lane)
    // and the tail's VOP-mode operands (224 a lane), and as one flip-flop (the card's ASIC point,
    // tm100: through 190 buffers to some 2,700 endpoints, its complement 10 buffers deep into
    // the tail's multiplier selects, the worst register path) only Vivado replicated it (its
    // max_fanout). Each copy is loaded with ss_act's next value, so it equals ss_act (cm_*: its
    // complement) in the same cycle, and is an instance of its own (kept: identical flip-flops
    // would be merged into one again). Without the streams (SE_SS 0) they are constants: VOP mode
    wire sa_n = !rst && ss_act_d;
    if (SE_SS) begin : g_ssc
      for (genvar l = 0; l < LANES; l++) begin : g_sar
        (* keep_hierarchy = "yes" *)
        otpu_delay #(.W(1), .N(1)) u_sa (.clk, .en(1'b1), .d(sa_n), .q(sa_r[l]));
        (* keep_hierarchy = "yes" *)
        otpu_delay #(.W(1), .N(1)) u_cu (.clk, .en(1'b1), .d(!sa_n), .q(cm_u[l]));
        (* keep_hierarchy = "yes" *)
        otpu_delay #(.W(1), .N(1)) u_cq (.clk, .en(1'b1), .d(!sa_n), .q(cm_q[l]));
      end
    end else begin : g_vop
      assign sa_r = '0;
      assign cm_u = '1;
      assign cm_q = '1;
    end
    // the fill's segment index, a copy per lane (s1_fi's next stage, kept instances): one register
    // drove every lane's buffer write addresses (r22's m8: s_fi, fanout 816, 7.1 ns of route)
    for (genvar l = 0; l < LANES; l++) begin : g_sfi
      (* keep_hierarchy = "yes" *)
      otpu_delay #(.W(5), .N(1)) u_fi (.clk, .en(1'b1), .d(s1_fi), .q(s_fi[l]));
    end
    // the tail's X meta, aligned with slot 0's pacc (as mt is for a reduction): u_vt's cap
    // (v in otpu_vdelay)
    localparam int SB = $bits(ss_meta_t);
    logic [SB-1:0] s_xp, s_mp;
    assign s_xp = s_xm;
    otpu_vdelay #(.N(TA)) u_smtv (.clk, .rst, .en(en_b), .d(s_xp[SB-1]), .q(s_mp[SB-1]));
    otpu_delay #(.W(SB - 1), .N(TA)) u_smt (.clk, .en(en_b), .d(s_xp[SB-2:0]), .q(s_mp[SB-2:0]));
    assign s_mt = s_mp;
    assign s_first = ss_act && s_xm.first;
    logic t_yv, t_ov, t_qc, t_qrl;
    logic [7:0] t_qsub;
    f32_t t_qd [LANES];
    otpu_se_tail #(.LANES(LANES), .TA(TA), .QD(1), .DSTEP2(DSTEP2)) u_tail (
      .clk, .rst, .init(s_init), .cfg(s_cfg), .fk(s_fk), .fi(s_fi), .fd(s_fd),
      .pe(sen), .in_v(s_in_v), .in_d(s_in_d), .xd(s_xd), .xa(s_xa), .xm(s_xm),
      .kv(root), .kv_v(root_v && !vt_q), .y_v(t_yv), .y_d(ss_y_d), .o_v(t_ov), .o_d(ss_o_d),
      .qp_cap(t_qc), .qp_row_last(t_qrl), .qp_sub(t_qsub), .qp_d(t_qd),
      .qo_v(root_v && vt_q), .qo_d(root_o),
      .cm_u, .cm_q, .cen(en_l), .u_sel(c_sel[1]), .u_a(c_sa[1]), .u_b(c_sb[1]), .u_c(c_sc[1]),
      .u_e(c_se[1]), .u_y(c_uy), .q_sel(c_sel[2]), .q_a(c_sa[2]), .q_b(c_sb[2]), .q_c(c_sc[2]),
      .q_y(c_qy));
    // Q's partials registered here (QD = 1: the tail pads its Q path for it), so no path
    // runs from the tail's adders into the tree's; the roots come out in window order, so
    // a FIFO of their kinds routes each to kv (A) or back to the tail (Q)
    always_ff @(posedge clk)
      if (rst) q_cap <= 1'b0;
      else if (en_b) q_cap <= ss_act && t_qc;
    always_ff @(posedge clk) if (en_b) begin
      q_rl <= t_qrl;
      q_sub <= t_qsub;
      q_pd <= t_qd;
    end
    logic [7:0] kq;
    logic [2:0] kh, kt;
    always_ff @(posedge clk)
      if (rst || s_init) begin
        kh <= '0; kt <= '0;
      end else if (en_b) begin
        if (cap && vt_rl) begin
          kq[kt] <= q_cap;
          kt <= kt + 1'b1;
        end
        if (root_v) kh <= kh + 1'b1;
      end
    assign vt_q = kq[kh];
    assign ss_y_v = sen && t_yv;
    assign ss_o_v = sen && t_ov;
  end else begin : g_nse
    assign ss_rq = 1'b0;
    assign ss_act_d = 1'b0;
    assign q_cap = 1'b0;
    assign q_rl = 1'b0;
    assign q_sub = '0;
    assign vt_q = 1'b0;
    assign ss_act = 1'b0;
    assign ss_gnt = 1'b0;
    assign sa_r = '0;
    assign cm_u = '1;
    assign cm_q = '1;
    assign sen = 1'b0;
    assign s_init = 1'b0;
    assign s_first = 1'b0;
    assign s_xm = '0;
    assign s_mt = '0;
    for (genvar l = 0; l < LANES; l++) begin : g_z
      assign s_xd[l] = '0;
      assign s_xa[l] = '0;
      assign ss_y_d[l] = '0;
      assign q_pd[l] = '0;
      assign c_uy[l] = '0;
      assign c_qy[l] = '0;
    end
    assign ss_y_v = 1'b0;
    assign ss_o_v = 1'b0;
    assign ss_o_d = '0;
  end

  // ------------------------------------------------------------------ FOLLOW
  // cqf: the queued instructions' fol (cleared when the collective completes); f_cur: the
  // address of the A chunk the front forms next (the head's A before it begins), f_end: A's
  // end (one row); f_hv: cycles the head has waited not issuing (2: f_chk is its A's). A chunk
  // whose read is taken at cycle p was formed at p - 1 and enabled by f_ok from cycle p - 4 at
  // the latest (f_top's register, then SE's en_q, en_r), so at most three chunks are formed
  // past the f_cur it was decided on: FK words, read only below the top of writes granted
  // at least two cycles before (otpu_slice's f_top is one register, f_top here a second)
  localparam int FK = 4 * LANES;
  if (FOLLOW) begin : g_fol
    logic [1:0]    cqf, f_hv;
    logic [AW-1:0] f_cur, f_end, f_top;
    logic          f_act, f_chk;
    wire           fc = (f_cur + AW'(FK) <= f_top) || (f_end <= f_top);
    assign f_wait = cqf[cq_h] && !(f_hv == 2'd2 && f_chk);
    always_ff @(posedge clk) begin
      f_top <= AW'(fol_top);
      if (!issuing) begin
        f_cur <= AW'(hc.w2);
        f_end <= AW'(hc.w2) + AW'(hc.w4[31:16]);
      end else if (en && !filling && iss_ok) f_cur <= f_cur + AW'(iwid);
      f_chk <= fc;
      if (rst) begin
        cqf <= '0; f_hv <= '0; f_act <= 1'b0; f_ok <= 1'b1;
      end else begin
        if (fol_done) cqf <= '0;
        if (start) cqf[cq_h ^ cq_n[0]] <= fol && !fol_done;
        f_hv <= (issuing || cq_n == 0 || beg_c) ? 2'd0 : (f_hv == 2'd2) ? 2'd2 : f_hv + 2'd1;
        if (beg_c) f_act <= cqf[cq_h] && !fol_done && hc.w4[15:0] == 16'd1;
        else if (fol_done) f_act <= 1'b0;
        f_ok <= !(f_act && issuing && !fol_done) || fc;
      end
    end
`ifndef SYNTHESIS
    // every A read taken while following is below the top of the GATHER's writes granted two
    // cycles before (fol_top a cycle late), and a follower is a one-row reduction of A alone
    logic [AW-1:0] f_g1;
    always_ff @(posedge clk) f_g1 <= AW'(fol_top);
    always @(posedge clk)
      if (!rst) begin
        if (beg_c && cqf[cq_h] && !fol_done &&
            (!red_f(hf) || hf == V_RDOT || hc.w4[15:0] != 16'd1))
          $fatal(1, "otpu_vpu: a follower (FOLLOW) that is not a one-row reduction of A");
        if (f_act && en)
          for (int l = 0; l < LANES; l++)
            if (ta_en[l] && AW'(ta_addr[l]) >= f_g1)
              $fatal(1, "otpu_vpu: FOLLOW reads %0d at or above the GATHER's top %0d", ta_addr[l],
                     f_g1);
      end
`endif
  end else begin : g_nfol
    assign f_ok = 1'b1;
    assign f_wait = 1'b0;
  end

  // ------------------------------------------------------------------ sequencing
  // The front: accept, begin, issue (on en). The back end: the completions (on en_b, FB: a cycle
  // behind), and the state a reduction keeps as it completes (initialized at its begin, which the
  // back end sees a cycle late: begr_b, with the front's registers loaded at the begin).
  logic [AW-1:0] wr_row_i;                    // the reduction's first row, its RDOT buffer
  logic          rdb_i;
  logic          begr_b;                      // a reduction began (empty_b: an empty one)
  wire           beg_c = can_begin && en && !f_wait;
  if (FB) begin : g_bcp
    always_ff @(posedge clk) begin
      func_b <= func; tag_b <= tag; imm_b <= imm; rows_b <= rows; drs_b <= drs; dst_b <= dst;
      od_one_b <= od_one; od_sc_b <= od_sc; st_frz_b <= st_frz; st_c_b <= st_c;
      begr_b <= !rst && beg_c && !h_empty && red_f(hf);
      begf_b <= !rst && beg_c && !h_empty && h_fs;
      empty_b <= !rst && beg_c && h_empty;
    end
    // (a cycle after the begin: its registers hold the instruction)
    assign wr_row_i = dst;
    assign rdb_i = (func == V_RDOT) && (rows <= 16'(NRB)) && (drs == 16'd1);
  end else begin : g_bnc
    assign func_b = func; assign tag_b = tag; assign imm_b = imm; assign rows_b = rows;
    assign drs_b = drs; assign dst_b = dst; assign od_one_b = od_one; assign od_sc_b = od_sc;
    assign st_frz_b = st_frz; assign st_c_b = st_c;
    assign begr_b = beg_c && !h_empty && red_f(hf);
    assign begf_b = beg_c && !h_empty && h_fs;
    assign empty_b = beg_c && h_empty;
    assign wr_row_i = AW'(hc.w1);
    assign rdb_i = (hf == V_RDOT) && (hc.w4[15:0] <= 16'(NRB)) && (hc.w5[15:0] == 16'd1);
  end
  // the completions this cycle: an elementwise instruction's last chunk is written (an FSUM's
  // too); a reduction's last row (RMAX), pair (ARGMAX), row sum or flush (FSUM's: no done)
  logic fin_c, ewfin_c;
  always_comb begin
    ewfin_c = en_b && mo.v && mo.all_last;
    fin_c = 1'b0;
    if (en_b) begin
      if (red_act_b && is_mx_b && live(mxm_q, tag_b) && mxm_q.all_last && !is_am_b) fin_c = 1'b1;
      if (red_act_b && is_am_b && am4_v && am4_l) fin_c = 1'b1;
      if (red_act_b && is_sum_b && root_v && rows_done + 1 == rows_b && !rdb) fin_c = 1'b1;
      if (fl_w && 32'(fl) + 32'(LANES) >= 32'(fl_rows)) fin_c = 1'b1;
    end
  end

  // the front
  always_ff @(posedge clk) begin
    st_c <= busy && !en;
    st_frz <= st_frz + 32'(st_c);
    if (rst) begin
      cq_n <= '0; cq_h <= 1'b0;
      issuing <= 1'b0; red_act <= 1'b0; filling <= 1'b0; fs_act <= 1'b0; fs_i <= 1'b0;
      mac_i <= 1'b0;
      ew_n <= '0; last_tap <= '0;
      mi <= '0;
      tag <= '0;
    end else begin
      logic [6:0] ewn;
      logic [1:0] qn;
      ewn = ew_n;
      qn = cq_n;
      // ---- accept from the sequencer
      if (start) begin
        cq[cq_h ^ cq_n[0]] <= cmd;
        qn = qn + 1;
      end
      // ---- begin issuing the head instruction (an empty one completes: empty_b)
      if (beg_c) begin
        logic [15:0] c16, nchunks;
        cq_h <= ~cq_h;
        qn = qn - 1;
        if (!h_empty) begin
          dst <= AW'(hc.w1); a <= AW'(hc.w2); b <= AW'(hc.w3);
          rows <= hc.w4[15:0]; cols <= hc.w4[31:16];
          drs <= hc.w5[15:0]; ars <= hc.w5[31:16];
          brs <= hc.w6[15:0]; func <= hf; bmode <= hc.w6[25:24];
          imm <= hc.w7;
          nslots <= n_slots(hf);
          c16 = hc.w4[31:16];
`ifndef SYNTHESIS
          nchunks = comp_f(hf) ? (c16 + cwid - 1) / cwid : (c16 + 16'(LANES) - 1) >> LW;
`else
          nchunks = comp_f(hf) ? (c16 + 16'(NCL) - 1) >> CLW : (c16 + 16'(LANES) - 1) >> LW;
`endif
          if (hf == V_RSUM || hf == V_RSSQ || hf == V_RDOT)
            nchunks = ((nchunks + 16'(RL) - 1) / 16'(RL)) * 16'(RL);
          nch <= nchunks;
          ir <= '0; ic <= '0; ch <= '0;
          a_row <= AW'(hc.w2); b_row <= AW'(hc.w3); d_row <= AW'(hc.w1);
          // OUTER: A is dst (in place), after a fill of its column buffers; MAC: port B reads
          // dst (B_FULL's walk, at drs), after a fill of the C buffer with B (B_COL)
          filling <= (OXV && (hf == V_OUTER)) || (MAC && hf == V_MAC && hc.w6[25:24] == B_COL);
          nfill <= (nchunks < 16'd2) ? 16'd2 : nchunks;
          oc_a <= (MAC && hf == V_MAC) ? AW'(hc.w3) : AW'(hc.w7); od_a <= AW'(hc.w2);
          od_sc <= hc.flags[VF_DSCALAR]; od_one <= hc.flags[VF_DONE] || (MAC && hf == V_MAC);
          mac_i <= MAC && hf == V_MAC;
          if (hf == V_OUTER) begin
            a_row <= AW'(hc.w1); ars <= hc.w5[15:0];
          end
          if (MAC && hf == V_MAC) begin
            b_row <= AW'(hc.w1); brs <= hc.w5[15:0];
          end
          tag <= tag + 1;
          issuing <= 1'b1;
          fs_i <= h_fs;
          if (h_fs) begin
            fs_act <= 1'b1;
            fs_rows <= hc.w4[15:0];
            fs_imm <= hc.w7;
          end
          st_frz <= '0;
          if (red_f(hf)) red_act <= 1'b1;
          else begin
            ewn = ewn + 1;
            last_tap <= n_slots(hf);
          end
        end
      end
      if (en) begin
        // ---- issue the next chunk (its data arrives next cycle, described by mi)
        mi <= '0;
        if (issuing && filling) begin
          mi.fill <= 1'b1;
          mi.cb <= CBW'(ch);
          if (ch + 1 == nfill) begin
            filling <= 1'b0;
            ic <= '0; ch <= '0;
          end else begin
            ic <= ic + 16'(LANES);
            ch <= ch + 1;
          end
        end else if (issuing && iss_ok) begin
          mi.v <= 1'b1;
          mi.tag <= tag;
          mi.func <= func;
          mi.bmode <= bmode;
          mi.imm <= imm;
          mi.mask <= imask;
          mi.waddr <= is_red ? d_row : d_row + AW'(ic);
          mi.row_last <= irow_last;
          mi.all_last <= iall_last;
          mi.first <= (ch < 16'(RL));
          mi.final_ <= (ch + 16'(RL) >= nch);
          mi.sub <= 8'(ch % 16'(RL));
          mi.sq <= (func == V_RSSQ);
          mi.dot <= (func == V_RDOT);
          mi.cb <= CBW'(ch);
          mi.col <= ic;
          mi.fs <= fs_i;
          mi.par <= ir[0];
          if (irow_last) begin
            ic <= '0; ch <= '0;
            a_row <= a_row + AW'(ars);
            b_row <= b_row + AW'(brs);
            d_row <= d_row + AW'(drs);
            if (ir + 1 == rows) issuing <= 1'b0;
            else ir <= ir + 1;
          end else begin
            ic <= ic + iwid;
            ch <= ch + 1;
          end
        end
      end
      // ---- the back end's completions (FB: a cycle later than the pipeline's own)
      if (ewfin_c) ewn = ewn - 1;
      ew_n <= ewn;
      cq_n <= qn;
      if (fin_c) begin red_act <= 1'b0; fs_act <= 1'b0; end
    end
  end

  // the back end
  always_ff @(posedge clk) begin
    done_i <= 1'b0;
    pf_u <= 1'b0;
    if (rst) begin
      red_act_b <= 1'b0; flushing <= 1'b0; fs_b <= 1'b0;
    end else begin
      if (begf_b) begin
        fs_b <= 1'b1;
        rows_done <= '0;
        fl <= '0;
      end
      if (begr_b) begin
        red_act_b <= 1'b1;
        wr_row <= wr_row_i; rows_done <= '0;
        rdb <= rdb_i;
        fl <= '0;
        mx_have <= 1'b0;
      end
      if (en_b) begin
        // ---- RMAX
        if (red_act_b && is_mx_b && live(mxm_q, tag_b)) begin
          if (mxm_q.row_last) mx_have <= 1'b0;
          else begin
            mx_run <= mx_new;
            mx_ri <= mx_ni;
            mx_have <= 1'b1;
          end
        end
        // ---- RSUM/RSSQ/RDOT: a row's sum is written (or buffered) this cycle (see TMEM writes)
        if (red_act_b && is_sum_b && root_v) begin
          wr_row <= wr_row + AW'(drs_b);
          rows_done <= rows_done + 1;
          if (rows_done + 1 == rows_b && rdb) flushing <= 1'b1;
        end
        // ---- FSUM: a row's sum is buffered (the flush writes them after the last)
        if (fs_b && root_v) begin
          rows_done <= rows_done + 1;
          if (rows_done + 1 == fs_rows) flushing <= 1'b1;
        end
        // ---- RDOT (FSUM): the buffered row sums are written, LANES per cycle
        if (fl_w) begin
          fl <= fl + 16'(LANES);
          if (32'(fl) + 32'(LANES) >= 32'(fl_rows)) flushing <= 1'b0;
        end
      end
      if (fin_c) begin red_act_b <= 1'b0; fs_b <= 1'b0; end
      if (empty_b) done_i <= 1'b1;
      if ((fin_c && !fs_b) || ewfin_c) begin
        done_i <= 1'b1;
        pf_u <= 1'b1;
        pf_frz <= st_frz_b + 32'(st_c_b);
      end
    end
  end
`ifndef SYNTHESIS
  // an EXP2SUB or RSUM with flag FSUM on a VPU built without FSUM would run as one without it
  always @(posedge clk)
    if (!(FSUM && C8) && !rst && rst_seen && beg_c && hc.flags[VF_FSUM] &&
        (hf == V_EXP2SUB || hf == V_RSUM))
      $fatal(1, "otpu_vpu: VOP EXP2SUB / RSUM with FSUM on a VPU built without it (FSUM 0)");
  // FSUM's shapes (docs/isa.md VOP; the ISA simulator refuses the others)
  always @(posedge clk)
    if (!rst && beg_c && !h_empty && h_fs &&
        (hc.w4[15:0] > 16'(NRB) || hc.w4[31:16] < 16'(NP) || hc.w6[25:24] == B_SCALAR))
      $fatal(1, "otpu_vpu: FSUM on %0d x %0d (bmode %0d): rows <= %0d, cols >= %0d, B not a scalar",
             hc.w4[15:0], hc.w4[31:16], hc.w6[25:24], NRB, NP);
  // every take after an FSUM whose sums are not taken yet (the ISA simulator's rule)
  int fs_pn;
  always @(posedge clk)
    if (rst) fs_pn <= 0;
    else if (beg_c && h_take && fs_pn == 0) $fatal(1, "otpu_vpu: a take (RSUM FSUM) with no FSUM before it");
    else if (beg_c) fs_pn <= fs_pn + int'(h_fs && !h_empty) - int'(h_take);
  // the back end's red_act is the front's a cycle late but at the completions (FB)
  always @(posedge clk)
    if (FB && !rst && rst_seen && red_act_b && !red_act)
      $fatal(1, "otpu_vpu: the back end reduces with no reduction in the front");
`endif
endmodule
