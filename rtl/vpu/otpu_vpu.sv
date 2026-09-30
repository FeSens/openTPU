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
//   RECIP             6 (3 Newton steps)      RSQRT  10 (h = x/2, 3 Newton steps)
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
//             (passes x 25 cycles) orders the functions as their slot counts do.
// Without it (HAS_SE = 0 or LANES != 8) the composites run on the long lanes' slot chains.
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
  parameter bit HAS_SE = 1'b0                         // the stream engine (LANES = 8)
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
  output logic [LANES-1:0]        ta_en,
  output logic [LANES-1:0][31:0]  ta_addr,
  input  logic [LANES-1:0][31:0]  ta_data,
  output logic [LANES-1:0]        tb_en,
  output logic [LANES-1:0][31:0]  tb_addr,
  input  logic [LANES-1:0][31:0]  tb_data,
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
  logic [3:0]  nslots;                         // slots of this function
  logic [7:0]  tag;                            // instruction tag
  // cycles an instruction was frozen by the TMEM grant, for the profiler: the condition is
  // registered (st_c) and summed a cycle late, off the grant path; the count is st_frz + st_c
  // (an instruction begins on a granted cycle, so a begin never drops a pending frozen cycle
  // of its own)
  logic [31:0] st_frz;
  logic        st_c;

  function automatic logic [3:0] n_slots(input logic [7:0] f);
    case (f)
      V_ADD, V_SUB, V_RSUB, V_MUL, V_OUTER: return 4'd1;
      V_EXP2, V_EXP2SUB:           return 4'd9;
      V_RECIP:                     return 4'd6;
      V_RSQRT, V_LOG2:             return 4'd10;
      default:                     return 4'd0;
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
      default:                     return C_OTH;    // the reductions too
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

  // TMEM word address base + c + l, AW bits, zero-extended to the port
  function automatic logic [31:0] tm_addr(input logic [AW-1:0] base, input logic [15:0] c,
                                          input int l);
    logic [AW-1:0] t;
    t = base + AW'(c) + AW'(l);
    return 32'(t);
  endfunction
  wire [15:0] iwid = comp_f(func) ? 16'(NCL) : 16'(LANES);   // columns issued per cycle

  // C8: otpu_se_comp, whose stages are slot 0 (s = 0) and the tail's U (1) and Q (2). A
  // composite chunk is not issued while comp says slot 0 is taken when it would get there
  // (hold: HA = 2 cycles ahead, the read -> mi -> m0 path)
  localparam int CNS = 3;
  logic             c_hold, c_ov, c_s0;
  logic [CNS-1:0]   c_sel;
  f32_t             c_sa [CNS][LANES], c_sb [CNS][LANES], c_sc [CNS][LANES], c_se [CNS][LANES];
  f32_t             c_s0y [LANES], c_uy [LANES], c_qy [LANES];   // the stages' results
  f32_t             c_ia [LANES], c_ib [LANES], c_od [LANES];
  logic [LANES-1:0] c_om;
  logic [AW:0]      c_ometa;                   // {waddr, all_last}
  wire              iss_ok = !(C8 && comp_f(func) && c_hold);

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
  logic       s_init, s_in_v, s_first;
  f32_t       s_in_d [LANES], s_fd [LANES], s_xd [LANES], s_xa [LANES];
  logic [2:0] s_fk;
  logic [4:0] s_fi;
  ss_cfg_t    s_cfg;
  ss_meta_t   s_xm, s_mt;
  // the tail's Q final partials, registered here (q_*); the root is a Q dot's (vt_q)
  logic       q_cap, q_rl, vt_q;
  logic [7:0] q_sub;
  f32_t       q_pd [LANES];
  cmd_t hc;
  assign hc = cq[cq_h];
  wire  [7:0] hf = hc.w6[23:16];
  wire  h_empty = (hc.w4[15:0] == 0) || (hc.w4[31:16] == 0);
  wire  can_begin = (cq_n != 0) && !issuing && !red_act &&
                    ((red_f(hf) || h_empty) ? (ew_n == 0) :
                                              (ew_n == 0 || n_slots(hf) >= last_tap));
  wire  busy = issuing || red_act || ew_n != 0;


  // the pipeline enable: the grant, or (WBUF) the write buffer's room, registered. The enable
  // reaches every pipeline register of the lanes and the tree (~21k loads on the board), so it
  // must not come from the arbiter in the same cycle.
  logic en;
  assign ren = en;

  // ------------------------------------------------------------------ issue
  // The read addresses follow the mode alone (OUTER fill or not, B's mode): a lane's address
  // only counts while its enable is up, and a rotator port takes lane 0's whenever any lane
  // is, so the lane tests (ic + l < cols, the issue checks) reach the enables but not the
  // addresses, and TMEM's bank addresses no longer wait for them (133.33 MHz, fused-133
  // c2830d6: cols -> the lane test -> tb_en -> the address multiplexer -> port 4's bank rows
  // -> block RAM ADDRB, 9-10 levels, +0.130 ns)
  logic [LANES-1:0] imask;
  logic             irow_last, iall_last;
  always_comb begin
    ta_en = '0; tb_en = '0;
    imask = '0;
    irow_last = (ch + 1 == nch);
    iall_last = irow_last && (ir + 1 == rows);
    for (int l = 0; l < LANES; l++) begin
      if (filling) begin
        ta_addr[l] = od_sc ? 32'(od_a) : tm_addr(od_a, ic, l);
        tb_addr[l] = tm_addr(oc_a, ic, l);
      end else begin
        ta_addr[l] = tm_addr(a_row, ic, l);
        tb_addr[l] = (bmode == B_COL) ? tm_addr(b, ic, l) :
                     (bmode == B_ROW && l == 0) ? 32'(b_row) : tm_addr(b_row, ic, l);
      end
    end
    if (issuing && filling) begin
      // OUTER fill: C on port B; D on port A (one word on lane 0 for DSCALAR, none for DONE)
      for (int l = 0; l < LANES; l++) begin
        if (32'(ic) + 32'(l) < 32'(cols)) begin
          tb_en[l] = 1'b1;
          if (!od_one && (!od_sc || l == 0)) ta_en[l] = 1'b1;
        end
      end
    end else if (issuing && iss_ok) begin
      for (int l = 0; l < LANES; l++) begin
        if (32'(ic) + 32'(l) < 32'(cols) && 16'(l) < iwid) begin
          imask[l] = 1'b1;
          ta_en[l] = (func != V_FILL);
          if (reads_b(func) && (bmode == B_FULL || bmode == B_COL)) tb_en[l] = 1'b1;
        end
      end
      if (bmode == B_ROW && reads_b(func) && imask[0]) tb_en[0] = 1'b1;
    end
  end
`ifndef SYNTHESIS
  // an enabled lane's address is the one the lane tests chose
  always @(posedge clk)
    if (!rst)
      for (int l = 0; l < LANES; l++) begin
        logic [31:0] ra, rb;
        ra = filling ? (od_sc ? 32'(od_a) : tm_addr(od_a, ic, l)) : tm_addr(a_row, ic, l);
        rb = filling ? tm_addr(oc_a, ic, l) :
             (bmode == B_FULL) ? tm_addr(b_row, ic, l) :
             (bmode == B_COL) ? tm_addr(b, ic, l) : 32'(b_row);
        if ((ta_en[l] && ta_addr[l] != ra) || (tb_en[l] && tb_addr[l] != rb))
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
  } meta_t;
  // mi: the chunk read this cycle; m0: that chunk one granted cycle later, together with its
  // TMEM data registered (xa, xb), so no path runs from the TMEM block RAMs into the lanes'
  // multipliers in one cycle
  meta_t m0, mi;
  logic [LANES-1:0][31:0] xa, xb;
  f32_t xc [LANES], xd [LANES];               // OUTER: C(c), D(c) from the buffers, with xa/xb
  // stream mode: m0 is an RDOT with no chunk (slot 0 on the partial loop, fed by the tail)
  localparam meta_t M_SS = '{func: V_RDOT, default: '0};
  // m0's function decoded as it loads (m0d = fdec(m0.func)): the lanes' slot-0 operand and result
  // selects, tap 0's hit, the class and otpu_se_comp's entry come from flip-flops, not from
  // compares of m0.func (133.33 MHz, the 100 MHz fused placement: m0.func (fo 119) -> short lane
  // rc / rb, 13.6 / 13.8 levels, -1.2 ns; the tap-0 hit and the composite test fan out to every
  // lane's result and operand muxes)
  localparam int NVF = 20;                    // function codes V_ADD .. V_RDOT
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
    d.fh = foh(f);
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
    end else if (en) begin
      m0 <= (SE && ss_act) ? M_SS : mi;
      m0d <= fdec((SE && ss_act) ? M_SS.func : mi.func);
      xa <= ta_data;
      xb <= tb_data;
    end
`ifndef SYNTHESIS
  bit rst_seen;                               // (registers start arbitrary)
  initial rst_seen = 1'b0;
  always @(posedge clk)
    if (rst) rst_seen <= 1'b1;
    else if (rst_seen && m0d != fdec(m0.func)) $fatal(1, "otpu_vpu: m0d is not m0's func");
`endif
  // OUTER's column buffers (LUT RAM): written from the fill chunks' data at m0, read with the
  // main chunks' data (mi.cb) -- the fill runs >= 2 chunks ahead, so a word is written first
  for (genvar l = 0; l < LANES; l++) begin : g_cbuf
    f32_t cbuf [CBD], dbuf [CBD];
    always_ff @(posedge clk) if (en) begin
      if (m0.fill) begin
        cbuf[m0.cb] <= xb[l];
        dbuf[m0.cb] <= od_one ? F_ONE : od_sc ? xa[0] : xa[l];
      end
      xc[l] <= cbuf[mi.cb];
      xd[l] <= dbuf[mi.cb];
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
  // (EXP2 9: f v ii, RECIP 6: f k2, RSQRT 10: f k2, LOG2 10: f v) before being rewritten.
  // Derived from the slot programs in g_slot -- recheck when editing one. v is never live:
  // every slot followed by a read of v writes v itself. f is always live; dead fields are not
  // carried. LOG2 keeps t in k1 (slots 1..9), i2f(e) in k2 (slot 1's input stages to slot 9)
  // and e in ii (boundary 0 to slot 1's input stages).
  function automatic logic k1_live(input int s);
    return s <= 8;
  endfunction
  function automatic logic k2_live(input int s);
    return s <= 8;
  endfunction
  function automatic logic ii_live(input int s);
    return s <= 8;
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
  mt_t   msl [NSX];                          // meta at each slot's input mux
  for (genvar s = 0; s < NSX; s++) begin : g_mdel
    if (s == 1) begin : g_pre
      logic [MTW-1:0] q;
      otpu_delay #(.W(MTW), .N(PRE1)) u_p (.clk, .en, .d(mt_pk(mtap[s])), .q(q));
      assign msl[s] = mt_up(q);
    end else begin : g_nopre
      assign msl[s] = mtap[s];
    end
    if (s == 0) begin : g_d0
      otpu_delay #(.W($bits(mt_t)), .N(SL)) u_d (.clk, .en, .d(msl[s]), .q(mtap[s + 1]));
    end else begin : g_dn
      logic [MTW-1:0] q;
      otpu_delay #(.W(MTW), .N(SL)) u_d (.clk, .en, .d(mt_pk(msl[s])), .q(q));
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
        m0d.fh[V_RECIP]: begin
          st[0].k1 = {1'b1, ax[30:0]};                       // -|x|
          st[0].k2 = ftz(RECIP_MAGIC - ax);                  // seed
          st[0].f  = {xz[31], (ax >= 32'h7E80_0000), (ax == 0)};
        end
        m0d.fh[V_RSQRT]: begin
          st[0].k2 = RSQRT_MAGIC - (xz >> 1);
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
        always_ff @(posedge clk) if (en) begin
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
          sf[V_OUTER]: if (s == 0) begin ia = x; ib = xd[l]; ic_ = y; ie = xc[l]; dest = 2'd1; end
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
            if (s % 2 == 0) begin ia = t.k1; ib = t.k2; ic_ = F_TWO; dest = 2'd1; end   // 2 - |x|y
            else begin ia = t.k2; ib = t.v; dest = 2'd3; end                              // y = y*t
          end
          sf[V_RSQRT]: begin
            if (s == 0) begin ia = F_HALF; ib = ftz(x); dest = 2'd2; negd = 1'b1; end     // -h
            else if ((s - 1) % 3 == 0) begin ia = t.k2; ib = t.k2; dest = 2'd1; end       // y*y
            else if ((s - 1) % 3 == 1) begin ia = t.k1; ib = t.v; ic_ = F_1P5; dest = 2'd1; end
            else begin ia = t.k2; ib = t.v; dest = 2'd3; end                              // y*c
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
      always_ff @(posedge clk) if (en) begin
        ra <= ia; rb <= ib; sr <= sin_; dr <= dest; nr <= negd;
      end
      // rc has a sync reset so it can't join u_c's two stages in a shift-register LUT: the
      // adder's c operand then leaves a flip-flop (clk->Q 0.3 ns) instead of an SRL (1.5 ns).
      // F_NZ is every slot's default c, so bits constant across a slot's cases stay constant.
      always_ff @(posedge clk)
        if (rst)     rc <= F_NZ;
        else if (en) rc <= ic_;
      if (s == 0) begin : g_mma
        // slot 0: a*b + c*e (OUTER's second product; e = 1.0 otherwise)
        f32_t re;
        always_ff @(posedge clk) if (en) re <= ie;
        otpu_fmma #(.LM(LM), .LA(LA)) u_ma (.clk, .en, .a(ra), .b(rb), .c(rc), .e(re), .y(r));
      end else begin : g_ma
        otpu_fmadd #(.LM(LM), .LA(LA)) u_ma (.clk, .en, .a(ra), .b(rb), .c(rc), .y(r));
      end
      if (s == 0 && RMA) begin : g_pacc
        assign pacc[l] = r;
      end
      // the live fields only (see k1_live); the dead ones read as 0 and are never used
      assign sd.v = '0;
      if (k1_live(s)) begin : g_k1
        otpu_delay #(.W(32), .N(LM + LA)) u_k1 (.clk, .en, .d(sr.k1), .q(sd.k1));
      end else begin : g_nk1
        assign sd.k1 = '0;
      end
      if (k2_live(s)) begin : g_k2
        otpu_delay #(.W(32), .N(LM + LA)) u_k2 (.clk, .en, .d(sr.k2), .q(sd.k2));
      end else begin : g_nk2
        assign sd.k2 = '0;
      end
      if (ii_live(s)) begin : g_ii
        otpu_delay #(.W(9), .N(LM + LA)) u_ii (.clk, .en, .d(sr.ii), .q(sd.ii));
      end else begin : g_nii
        assign sd.ii = '0;
      end
      otpu_delay #(.W(3), .N(LM + LA)) u_f (.clk, .en, .d(sr.f), .q(sd.f));
      otpu_delay #(.W(3), .N(LM + LA)) u_dst (.clk, .en, .d({dr, nr}), .q({dd, nd}));
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
    assign rc_r = tc.f[0] ? F_ZERO : tc.f[1] ? {tc.f[2], 31'd0} : (tc.f[2] ? fneg(tc.k2) : tc.k2);
    assign rs_r = ts.f[0] ? F_ZERO : ts.k2;
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
        m0d.fh[V_OUTER]: begin ia = x; ib = xd[l]; ic_ = y; ie = xc[l]; end
        m0d.fh[V_ADD]:  begin ia = x; ic_ = y; end
        m0d.fh[V_SUB]:  begin ia = x; ic_ = fneg(y); end
        m0d.fh[V_RSUB]: begin ia = y; ic_ = fneg(x); end
        m0d.fh[V_RSUM], m0d.fh[V_RSSQ], m0d.fh[V_RDOT]:
          if (RMA) begin ia = rsa[l]; ib = rsb[l]; ic_ = rfb[l]; end
        default: ;
      endcase
    end
    always_ff @(posedge clk) if (en) begin
      ra <= c_s0 ? c_sa[0][l] : ia; rb <= c_s0 ? c_sb[0][l] : ib; re <= c_s0 ? c_se[0][l] : ie;
    end
    // sync reset keeps rc out of u_c's SRL (see g_slot)
    always_ff @(posedge clk)
      if (rst)     rc <= F_NZ;
      else if (en) rc <= c_s0 ? c_sc[0][l] : ic_;
    otpu_fmma #(.LM(LM), .LA(LA)) u_ma (.clk, .en, .a(ra), .b(rb), .c(rc), .e(re), .y(st[1]));
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
    otpu_se_comp #(.LANES(LANES), .MW(AW + 1), .NS(CNS), .HA(2), .EXT(1'b1)) u_comp (
      .clk, .rst, .en, .in_v(m0.v && m0d.cf), .in_c(m0d.cc), .in_a(c_ia),
      .in_b(c_ib), .in_m(m0.mask), .in_meta({m0.waddr, m0.all_last}), .hold(c_hold),
      .st_sel(c_sel), .st_a(c_sa), .st_b(c_sb), .st_c(c_sc), .st_e(c_se), .st_y(sy),
      .out_v(c_ov), .out_d(c_od), .out_m(c_om), .out_meta(c_ometa));
    assign c_s0 = c_sel[0];
  end else begin : g_nocomp
    assign c_hold = 1'b0;
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
  always_ff @(posedge clk) if (en) begin
    f32_t v [LANES];
    logic h [LANES];
    logic up;
    for (int l = 0; l < LANES; l++) begin
      v[l] = ftz(xa[l]);
      h[l] = m0.mask[l];
    end
    for (int l = 0; l < HL; l++) begin
      up = !h[l] || (h[l + HL] && fp_gt(v[l + HL], v[l]));
      mxh_v[0][l] <= up ? v[l + HL] : v[l];
      mxh_h[0][l] <= h[l] || h[l + HL];
      mxh_i[0][l] <= up ? LW'(l + HL) : LW'(l);
    end
    mxh_m[0] <= m0r;
    mxh_c[0] <= m0.col;
    for (int j = 1; j <= ML; j++) begin
      for (int l = 0; l < (HL >> j); l++) begin
        up = !mxh_h[j-1][l] || (mxh_h[j-1][l + (HL >> j)] &&
                                fp_gt(mxh_v[j-1][l + (HL >> j)], mxh_v[j-1][l]));
        mxh_v[j][l] <= up ? mxh_v[j-1][l + (HL >> j)] : mxh_v[j-1][l];
        mxh_h[j][l] <= mxh_h[j-1][l] || mxh_h[j-1][l + (HL >> j)];
        mxh_i[j][l] <= up ? mxh_i[j-1][l + (HL >> j)] : mxh_i[j-1][l];
      end
      mxh_m[j] <= mxh_m[j-1];
      mxh_c[j] <= mxh_c[j-1];
    end
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
    end else if (en) begin
      am1_v <= red_act && is_am && live(mxm_q, tag) && mxm_q.row_last;
      am1_l <= mxm_q.all_last; am1_a <= mxm_q.waddr; am1_m <= mx_new; am1_i <= mx_ni;
      am2_v <= am1_v; am2_l <= am1_l; am2_a <= am1_a; am2_m <= am1_m;
      am2_s <= 32'(am1_i) + imm;
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
  rm_t   mtq, mt;                       // meta at the adder inputs / aligned with `pacc`
  otpu_delay #(.W($bits(rm_t)), .N(RMA ? LM + 1 : LM)) u_mtq (.clk, .en, .d(m0r), .q(mtq));
  otpu_delay #(.W($bits(rm_t)), .N(LA)) u_mt (.clk, .en, .d(mtq), .q(mt));
  logic  first_e;                       // !RMA: mtq.first one cycle early
  if (RMA) begin : g_nfe
    assign first_e = 1'b0;
  end else begin : g_fe
    otpu_delay #(.W(1), .N(LM - 1)) u_first (.clk, .en, .d(m0.first), .q(first_e));
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
    assign rsa[l] = (SE && ss_act) ? s_xd[l] : sa;
    assign rsb[l] = (SE && ss_act) ? s_xa[l] : sb;
    // pacc(chunk c) = pacc(chunk c - RL) + term(c): a loop of exactly RL cycles
    if (RMA) begin : g_ma
      // through slot 0's u_ma (term + prev: the add is commutative bit for bit), fed back by
      // fbq, which slot 0 reads (as c) the cycle after it's written: cleared on the chunks that
      // are on m0 then (mi), the first ones' +0. rc keeps the adder's c off a shift register.
      // A stream clears c instead, on its X segment's first flag (mi is idle then).
      f32_t fbq, fbd;
      otpu_delay #(.W(32), .N(DF - 1)) u_fb (.clk, .en, .d(pacc[l]), .q(fbd));
      always_ff @(posedge clk) if (en) fbq <= mi.first ? F_ZERO : fbd;
      assign rfb[l] = s_first ? F_ZERO : fbq;
    end else begin : g_acc
      f32_t tq, prev;
      assign rfb[l] = F_ZERO;
      otpu_fmul #(.LAT(LM)) u_sq (.clk, .en, .a(sa), .b(sb), .y(tq));
      if (RL - LA >= 1) begin : g_fbq
        // the loop's last stage is a flip-flop with a sync clear (the first chunks' +0): no
        // shift-register LUT and no mux in front of the adder
        f32_t fbq, fbd;
        otpu_delay #(.W(32), .N(RL - LA - 1)) u_fb (.clk, .en, .d(pacc[l]), .q(fbd));
        always_ff @(posedge clk) if (en) fbq <= first_e ? F_ZERO : fbd;
        assign prev = fbq;
      end else begin : g_fbw
        f32_t fb;
        otpu_delay #(.W(32), .N(RL - LA)) u_fb (.clk, .en, .d(pacc[l]), .q(fb));
        assign prev = mtq.first ? F_ZERO : fb;
      end
      otpu_fadd #(.LAT(LA)) u_acc (.clk, .en, .a(prev), .b(tq), .y(pacc[l]));
    end
  end

  // the folding tree (otpu_vtree): a row's sum RD cycles after its last final partial
  // a final partial of this reduction (or of the stream's dot A) is on `pacc`, or one of the
  // tail's dot Q, in a window of its own (q_cap)
  wire  a_cap = (SE && ss_act) ? s_mt.v && s_mt.final_ :
                                 red_act && is_sum && live(mt, tag) && mt.final_;
  wire  cap = a_cap || q_cap;
  wire  vt_rl = q_cap ? q_rl : (SE && ss_act) ? s_mt.row_last : mt.row_last;
  wire  [7:0] vt_sub = q_cap ? q_sub : (SE && ss_act) ? s_mt.sub : mt.sub;
  f32_t vt_in [LANES];
  for (genvar l = 0; l < LANES; l++) begin : g_vtin
    assign vt_in[l] = q_cap ? q_pd[l] : pacc[l];
  end
  // root_o: the root from a copy of its register, for the DMA's O path alone (the tail's Q dot,
  // qo_d -> o_d -> ss_o_d): with no load here it can sit between the VPU and the DMA, so the
  // wire to the DMA's o_q spans two cycles (133.33 MHz, 110ec6d: u_vt root -> u_dma o_q, 0
  // levels, 6.6 ns of route, +0.102 ns)
  f32_t root, root_o;
  logic root_v;
  otpu_vtree #(.LANES(LANES), .LA(LA)) u_vt (.clk, .rst(rst || s_init), .en, .pacc(vt_in), .cap,
                                              .row_last(vt_rl), .sub(vt_sub), .root, .root_o,
                                              .root_v);
`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && en && a_cap && q_cap) $fatal(1, "otpu_vpu: A and Q windows meet on u_vt");
  always_ff @(posedge clk)
    if (root_o != root) $fatal(1, "otpu_vpu: root_o %h is not root %h", root_o, root);
`endif

  // rows finish in order: the next root goes to wr_row
  logic [AW-1:0] wr_row;
  logic [15:0]   rows_done;
  // RDOT's row-sum buffer: row i in bank i % LANES; the flush writes rows fl .. fl+LANES-1
  localparam int NRB = 256;
  logic          rdb;                        // this reduction's row sums are buffered
  logic          flushing;
  logic [15:0]   fl;
  f32_t          rbuf_q [LANES];
  for (genvar l = 0; l < LANES; l++) begin : g_rbuf
    // distributed RAM: the flush reads a row sum the cycle after its write (the last row's).
    // With the write buffer (WBUF) behind the read, Vivado made this a block RAM with fl as its
    // read-address register, which returns the old word on that collision: every RDOT of at
    // most LANES rows then wrote its last row sum stale (on the card, not in simulation)
    (* ram_style = "distributed" *) f32_t rb [NRB / LANES];
    always_ff @(posedge clk)
      if (en && red_act && is_sum && root_v && rdb && rows_done[LW-1:0] == LW'(l))
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
  for (genvar l = 0; l < LANES; l++) begin : g_twa
    assign tw_addr[l] = 32'(tw_a[l]);
  end
  logic                     wb_e;          // no write pending
  if (!WBUF) begin : g_wdir
    assign en = (SE && ss_act) ? sen : gnt;
    assign wb_e = (tw_en == '0);
    always_ff @(posedge clk) begin
      if (rst) begin
        tw_en <= '0;
        done <= 1'b0; dpend <= 1'b0;
      end else begin
        if (gnt) begin
          tw_en <= cw_en; tw_a <= cw_addr; tw_data <= cw_data;
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
    localparam int NWQ = SE ? 4 : 2;
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
    wire  push = en && ((|cw_en) || pdn);
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
          en_q <= ss_act_d ? ss_pe : pop ? (wn_p1 < 2) : (wn_p0 < 2);  // a stream: its pe (as sen)
          en_r <= en_q;
        end else begin
          en_r <= pop ? (wn_p1 < 2) : (wn_p0 < 2);
        end
        done <= pop && hd.dn;
        dpend <= pdn && !en;
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
    if (red_act && func == V_RMAX && live(mxm_q, tag) && mxm_q.row_last) begin
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
    if (red_act && is_sum && root_v && !rdb) begin
      cw_en[0] = 1'b1;
      cw_addr[0] = wr_row;
      cw_data[0] = root;
      mm[0] = 1'b0;
    end
    if (flushing) begin
      for (int l = 0; l < LANES; l++) begin
        if (32'(fl) + 32'(l) < 32'(rows)) begin
          cw_en[l] = 1'b1;
          cw_addr[l] = dst + AW'(fl) + AW'(l);
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
  // drained.
  if (SE) begin : g_se
    wire idle = !start && (cq_n == 0) && !busy && !done_i && !dpend && wb_e;
    assign ss_act_d = ss_req && ss_rq && (ss_act || idle);
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
        ss_rq <= ss_req;
        ss_act <= ss_act_d;
        s_init <= ss_req && ss_rq && !ss_act && idle;
        sen_q <= ss_act_d && ss_pe;
        sen <= sen_q;
        s1_fk <= ss_act ? ss_fk : SF_NONE;
        s_fk <= s1_fk;
      end
      s1_fi <= ss_fi; s1_fd <= ss_fd; s_fi <= s1_fi; s_fd <= s1_fd;
      s1_in_v <= ss_in_v; s1_in_d <= ss_in_d; s_in_v <= s1_in_v; s_in_d <= s1_in_d;
      s_cfg <= ss_cfg;
    end
    assign ss_gnt = ss_act;
    // the tail's X meta, aligned with slot 0's pacc (as mt is for a reduction): u_vt's cap
    otpu_delay #(.W($bits(ss_meta_t)), .N(TA)) u_smt (.clk, .en, .d(s_xm), .q(s_mt));
    assign s_first = ss_act && s_xm.first;
    logic t_yv, t_ov, t_qc, t_qrl;
    logic [7:0] t_qsub;
    f32_t t_qd [LANES];
    otpu_se_tail #(.LANES(LANES), .TA(TA), .QD(1)) u_tail (
      .clk, .rst, .init(s_init), .cfg(s_cfg), .fk(s_fk), .fi(s_fi), .fd(s_fd),
      .pe(sen), .in_v(s_in_v), .in_d(s_in_d), .xd(s_xd), .xa(s_xa), .xm(s_xm),
      .kv(root), .kv_v(root_v && !vt_q), .y_v(t_yv), .y_d(ss_y_d), .o_v(t_ov), .o_d(ss_o_d),
      .qp_cap(t_qc), .qp_row_last(t_qrl), .qp_sub(t_qsub), .qp_d(t_qd),
      .qo_v(root_v && vt_q), .qo_d(root_o),
      .cm(!ss_act), .cen(en), .u_sel(c_sel[1]), .u_a(c_sa[1]), .u_b(c_sb[1]), .u_c(c_sc[1]),
      .u_e(c_se[1]), .u_y(c_uy), .q_sel(c_sel[2]), .q_a(c_sa[2]), .q_b(c_sb[2]), .q_c(c_sc[2]),
      .q_y(c_qy));
    // Q's partials registered here (QD = 1: the tail pads its Q path for it), so no path
    // runs from the tail's adders into the tree's; the roots come out in window order, so
    // a FIFO of their kinds routes each to kv (A) or back to the tail (Q)
    always_ff @(posedge clk) if (en) begin
      q_cap <= ss_act && t_qc;
      q_rl <= t_qrl;
      q_sub <= t_qsub;
      q_pd <= t_qd;
    end
    logic [7:0] kq;
    logic [2:0] kh, kt;
    always_ff @(posedge clk)
      if (rst || s_init) begin
        kh <= '0; kt <= '0;
      end else if (en) begin
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

  // ------------------------------------------------------------------ sequencing
  always_ff @(posedge clk) begin
    logic fin, ewfin;
    done_i <= 1'b0;
    pf_u <= 1'b0;
    fin = 1'b0;
    ewfin = 1'b0;
    st_c <= busy && !en;
    st_frz <= st_frz + 32'(st_c);
    if (rst) begin
      cq_n <= '0; cq_h <= 1'b0;
      issuing <= 1'b0; red_act <= 1'b0; filling <= 1'b0; flushing <= 1'b0;
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
      // ---- begin issuing the head instruction
      if (can_begin && en) begin
        logic [15:0] c16, nchunks;
        cq_h <= ~cq_h;
        qn = qn - 1;
        if (h_empty) begin
          done_i <= 1'b1;
        end else begin
          dst <= AW'(hc.w1); a <= AW'(hc.w2); b <= AW'(hc.w3);
          rows <= hc.w4[15:0]; cols <= hc.w4[31:16];
          drs <= hc.w5[15:0]; ars <= hc.w5[31:16];
          brs <= hc.w6[15:0]; func <= hf; bmode <= hc.w6[25:24];
          imm <= hc.w7;
          nslots <= n_slots(hf);
          c16 = hc.w4[31:16];
          nchunks = comp_f(hf) ? (c16 + 16'(NCL) - 1) >> CLW : (c16 + 16'(LANES) - 1) >> LW;
          if (hf == V_RSUM || hf == V_RSSQ || hf == V_RDOT)
            nchunks = ((nchunks + 16'(RL) - 1) / 16'(RL)) * 16'(RL);
          nch <= nchunks;
          ir <= '0; ic <= '0; ch <= '0;
          a_row <= AW'(hc.w2); b_row <= AW'(hc.w3); d_row <= AW'(hc.w1);
          // OUTER: A is dst (in place), after a fill of its column buffers
          filling <= (hf == V_OUTER);
          nfill <= (nchunks < 16'd2) ? 16'd2 : nchunks;
          oc_a <= AW'(hc.w7); od_a <= AW'(hc.w2);
          od_sc <= hc.flags[VF_DSCALAR]; od_one <= hc.flags[VF_DONE];
          if (hf == V_OUTER) begin
            a_row <= AW'(hc.w1); ars <= hc.w5[15:0];
          end
          tag <= tag + 1;
          issuing <= 1'b1;
          st_frz <= '0;
          if (red_f(hf)) begin
            red_act <= 1'b1;
            wr_row <= AW'(hc.w1); rows_done <= '0;
            rdb <= (hf == V_RDOT) && (hc.w4[15:0] <= 16'(NRB)) && (hc.w5[15:0] == 16'd1);
            fl <= '0;
            mx_have <= 1'b0;
          end else begin
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
        // ---- elementwise: an instruction's last chunk is written
        if (mo.v && mo.all_last) begin
          ewfin = 1'b1;
          ewn = ewn - 1;
        end
        // ---- RMAX
        if (red_act && is_mx && live(mxm_q, tag)) begin
          if (mxm_q.row_last) mx_have <= 1'b0;
          else begin
            mx_run <= mx_new;
            mx_ri <= mx_ni;
            mx_have <= 1'b1;
          end
          if (mxm_q.all_last && !is_am) fin = 1'b1;
        end
        if (red_act && is_am && am4_v && am4_l) fin = 1'b1;     // its last pair is written
        // ---- RSUM/RSSQ/RDOT: a row's sum is written (or buffered) this cycle (see TMEM writes)
        if (red_act && is_sum && root_v) begin
          wr_row <= wr_row + AW'(drs);
          rows_done <= rows_done + 1;
          if (rows_done + 1 == rows) begin
            if (rdb) flushing <= 1'b1;
            else fin = 1'b1;
          end
        end
        // ---- RDOT: the buffered row sums are written, LANES per cycle
        if (flushing) begin
          fl <= fl + 16'(LANES);
          if (32'(fl) + 32'(LANES) >= 32'(rows)) begin
            flushing <= 1'b0;
            fin = 1'b1;
          end
        end
      end
      ew_n <= ewn;
      cq_n <= qn;
      if (fin) red_act <= 1'b0;
      if (fin || ewfin) begin
        done_i <= 1'b1;
        pf_u <= 1'b1;
        pf_frz <= st_frz + 32'(st_c);
      end
    end
  end
endmodule
