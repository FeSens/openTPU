// Sequencer: fetches and decodes one instruction per cycle. NOP, LI, ADDI and LOOP execute
// here; every other instruction has its register-relative fields resolved and is dispatched,
// in program order, into a window of WIN slots together with its memory footprint.
//
// Scoreboard: at dispatch, the new instruction records which in-flight instructions it
// conflicts with (RAW, WAR or WAW on any TMEM, DRAM or ACT RAM range). Each cycle every free
// unit starts its OLDEST instruction whose dependencies have all completed -- not necessarily
// its oldest instruction: an instruction that conflicts with an older one never becomes ready
// before it, so starting younger independent work early is always safe. Units therefore
// overlap freely (and fill each other's bubbles) while the results stay exactly those of
// sequential execution (docs/isa.md). Each unit completes in start order. HALT waits for the
// window to drain.
//
// The MXU is special in two ways: it starts its instructions strictly in order, and in two
// phases. It may start streaming an MM from DRAM as soon as the MM's DRAM dependencies are
// clear (ustart), and it may consume the stream once all dependencies are clear (urel). Deep
// prefetch therefore overlaps the stream with whatever produces the MM's stationary operand.
//
// RLD (docs/isa.md) is dispatched like a unit instruction (to the collective unit, whose slice
// reads the TMEM word itself); R then holds the next instruction until the value comes back
// (rl_v): younger instructions may use R[rd] in their addresses and LOOP counts. The hold
// (rl_wait) is a flip-flop, so the fetch-enable path only gains one registered term. HALT with
// flag CHAIN reports R[ra] (the program's DRAM address) and R[rb] (its instructions) with the
// halt (ch_*): the slice reloads IMEM from there and restarts (otpu_slice).
//
// IMEM is a synchronous RAM (FPGA block RAM) of rows of IPR = D/32 instructions -- one DRAM
// chunk, so the slice's loader writes one row per cycle. It is read one instruction ahead, at
// fa: the exact pc that R moves to next (computed from flip-flops only). When R moves on, the
// instruction at fa is registered into ir, so no R-stage path starts at the RAM output. This
// costs one cycle at the program start (ir is empty after reset).
module otpu_seq
  import otpu_pkg::*;
#(
  parameter int IMEM_WORDS = 1 << 16,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int SID        = 0,
  parameter int S          = 1,
  parameter int D          = 32,
  parameter int WIN        = 16,
  parameter bit STREAMS    = 1'b1,    // the stream engine (otpu_slice HAS_SS); 0: DSTEP and STREAM
                                      //   are illegal instructions (ERROR), not a wait for SE
  // EXPERIMENTAL (docs/seq_vpu.md, docs/isa.md "Wait counts"): WAITC 1 takes the dependencies
  // from a wait entry the compiler writes beside each instruction instead of comparing
  // footprints: per in-order unit (DMA, MXU, QUANT, COLL) a 4-bit count k (0: no wait, else
  // until the unit's k-th last instruction before this one has completed; and the MM's stream
  // start's k for the DMA and the quantizer), and for the VPU, which starts its oldest ready
  // instruction, a mask of the VPU instructions 1..31 back it waits for. No footprints, no range
  // compares. 0: today's scoreboard.
  parameter bit WAITC      = 1'b0
) (
  input  logic                clk,
  input  logic                rst,
  output cmd_t                ucmd   [NUNITS],
  output logic [NUNITS-1:0]   ustart,
  output logic                urel,        // MXU: release the oldest started, unreleased MM
  input  logic [NUNITS-1:0]   urdy,
  input  logic [NUNITS-1:0]   udone,
  output logic                halted,
  output logic                error,
  output logic [31:0]         icount,
  output seq_ev_t             ev,          // trace and activity events, a cycle late (otpu_pkg)
  input  logic [31:0]         rinit  [8],  // R8..R15 while in reset: the run's arguments
  input  logic                rl_v,        // RLD's value is back: R[rd] = rl_val
  input  logic [31:0]         rl_val,
  output logic                ch_req,      // the halt is a HALT CHAIN: to ch_addr, ch_n
  output logic [31:0]         ch_addr,
  output logic [31:0]         ch_n,
  // IMEM write port (the loader; used while the slice is held in reset)
  input  logic                im_we,
  input  logic [31:0]         im_row,
  input  logic [D*8-1:0]      im_data,
  // WAITC: the wait entries' write port (the loader: a row of 4 64-bit entries as they are in
  // DRAM, instructions 4 * sc_row .. 4 * sc_row + 3)
  input  logic                sc_we,
  input  logic [31:0]         sc_row,
  input  logic [255:0]        sc_data
);
  localparam int SW = $clog2(WIN);
  localparam int IPR = D / 32;                   // instructions per IMEM row
  localparam int NROW = IMEM_WORDS / (8 * IPR);
  localparam int RW = $clog2(NROW);
  initial if (D % 32 != 0) $fatal(1, "otpu_seq: D must be a multiple of 32");
  initial if (DAW < 32 || DAW > 36) $fatal(1, "otpu_seq: DAW %0d (OTPU_DAW) is 32 .. 36", DAW);
  logic [D*8-1:0] irow;           // the RAM output: the row holding fa
  logic [255:0]   ir;             // the instruction at pc
  logic           ir_v;           // ir is valid (0 only in the first cycle after reset)
  logic [31:0]    fa, fa_d;       // the pc R moves to next; its next value
  logic [31:0]    fa_n;           // fa's next value when it moves (the RAM address)
  logic           fe;             // fa moves (the RAM's read enable; otherwise it keeps fa's row)
  logic           ld;             // R moves on to the instruction at fa

  logic [31:0] pc, cyc;
  logic [31:0] R [16];
  logic        stopping;         // HALT fetched: wait for the window to drain
  logic        rl_wait;          // an RLD is out: R holds until its value is back
  logic [3:0]  rl_rd;

  // loop stack
  logic [2:0]  sp;
  logic [31:0] stk_start [4];
  logic [31:0] stk_end   [4];
  logic [31:0] stk_rem   [4];
  // ... its top two entries (stk[sp-1], stk[sp-2]) also in flip-flops, with their rem > 1 / > 2
  // flags, and at_end (pc ends the top loop) registered as R moves on (250 MHz, FK35: sp -> the
  // stack's LUT RAM -> stk_end == fa -> the fetch address, 9 levels, -0.86 ns). The arrays stay
  // the stack's record for the entries below.
  logic [31:0] t_start, t_end, t_rem, n_start, n_end, n_rem;
  logic        t_gt1, t_gt2, n_gt1;
  logic        at_end_q;
  // LOOP takes three cycles: the first registers the trip count, the body length and the skip
  // target (pc holds), the second fetches the target (enter: pc + 1, skip: pc + 1 + len) from
  // registers and registers the count's flags, the third enters or skips the loop -- keeps the
  // register read and the adds off the fetch-address path (250 MHz, FK35: ir -> R[ra] -> the
  // trip count's zero test -> the fetch address, 10 levels, -0.87 ns)
  logic [1:0]  lp;
  logic [31:0] lp_cnt, lp_len, lp_skip;
  logic        lp_nz, lp_gt1, lp_gt2, lp_len1;

  // The footprint as the window keeps it: ranges stored by space, so ranges in different
  // spaces, and two reads, are never compared. No valid bits: an invalid range has hi = 0,
  // which overlaps nothing (b.lo < 0 is false) -- exactly ov()'s v terms. Routing (fp_seg):
  //   LD:     d[0] = rd0          t0 = wr0 W
  //   ST:     d[0] = wr0 (dw)     t0 = rd0
  //   DSTEP:  d[0] = wr0 (dw)     t0 = wr1 W, t1 = rd0, t2 = rd1, t3 = rd2
  //   STREAM: d = wr0, wr2 (dw)   t0 = wr1 W, t1 = rd0, t2 = rd1, t3 = rd2, t4 = rd3
  //   MM:     d = rd0, rd1        t0 = hull(wr0, wr1) W (RMAX), t2 = rd3 (ASCALE), a = rd2
  //   QACT:                       t0 = rd0, t1 = rd1 (CSCALE), t2 = rd2 (RSCALE), a = wr0 W
  //   QST:    d = wr0, wr1 (dw)   t0 = rd0
  //   VOP:                        t0 = wr0 W, t1 = rd0, t2 = rd1, t3 = rd2 (OUTER)
  //   GATHER:                     t0 = wr0 W, t1 = rd0
  // Only t0 may be written, so a footprint pair has 9 TMEM range pairs to compare. MM's RMAX
  // (the row maxima, the spare row after the output tile) is folded into its output range: the
  // hull [min lo, max hi) covers both, so every dependency on either is kept (ovr() is monotone
  // in the range); it adds only the output tile's padding words between them (ors - N).
  // An instruction's DRAM ranges are all reads or all writes, hence one dw bit. Their ends are
  // DAW bits, {REG, 32 bits} (otpu_pkg mkr): ranges of different regions never overlap, and
  // WAITW's one range [0, 2^DAW - 1] is every region's. ACT ranges
  // start below 2^8 (an 8-bit field) and are at most 2^16 - 1 long, so [alo, ahi) is exact.
  // TMEM ranges keep TAW/TAW+1 bits: TMEM holds 2^16 words (more in a configuration with a
  // larger TMEM: TAW follows it), so every access the ISA allows has lo < 2^TAW, hi <= 2^TAW.
  // A valid TMEM range that does not fit raises all instead (the instruction then conflicts
  // with everything, like BAR) -- conservative, never wrong.
  localparam int TAW = (TMEM_WORDS > (1 << 16)) ? $clog2(TMEM_WORDS) : 16;
  typedef struct packed { logic [DAW-1:0] lo, hi; } r32_t;     // DRAM (DAW bits: the name is old)
  typedef struct packed { logic [TAW-1:0] lo; logic [TAW:0] hi; } rt_t;
  typedef struct packed { logic w; rt_t r; } rtw_t;     // TMEM, may be written
  typedef struct packed {
    logic            all;
    logic            dw;        // the DRAM ranges are writes
    r32_t  [1:0]     d;         // DRAM
    rtw_t            t0;        // TMEM, the only range that may be written
    rt_t             t1;        // TMEM, always a read
    rt_t             t2;        // TMEM, always a read
    rt_t             t3;        // TMEM, always a read
    rt_t             t4;        // TMEM, always a read (STREAM's constants)
    logic            aw;        // ACT
    logic [7:0]      alo;
    logic [16:0]     ahi;
  } fps_t;

  function automatic r32_t r32(input rng_t x);
    r32_t o;
    o.lo = x.lo; o.hi = x.v ? x.hi : '0;
    return o;
  endfunction

  function automatic rt_t rt(input rng_t x);
    rt_t o;
    o.lo = x.lo[TAW-1:0]; o.hi = x.v ? x.hi[TAW:0] : '0;
    return o;
  endfunction

  // the TMEM range does not fit rt_t
  function automatic logic tovf(input rng_t x);
    return x.v && (x.lo[31:TAW] != '0 || x.hi[31:TAW+1] != '0);
  endfunction

  // the smallest range covering a and b (an invalid one, hi = 0, covers nothing)
  function automatic rt_t hull(input rt_t a, input rt_t b);
    rt_t o;
    if (b.hi == '0) o = a;
    else if (a.hi == '0) o = b;
    else begin
      o.lo = (b.lo < a.lo) ? b.lo : a.lo;
      o.hi = (b.hi > a.hi) ? b.hi : a.hi;
    end
    return o;
  endfunction

  function automatic fps_t fp_seg(input logic [7:0] op, input fp_t f);
    fps_t s;
    rng_t a, t0, t1, t2, t3, t4, h;  // ACT; TMEM t0, t1, t2, t3, t4; MM's RMAX (into t0)
    logic w0;
    s = '0; a = '0; t0 = '0; t1 = '0; t2 = '0; t3 = '0; t4 = '0; h = '0; w0 = 1'b0;
    case (op)
      OP_LD: begin
        s.d[0] = r32(f.rd[0]); t0 = f.wr[0]; w0 = 1'b1;
      end
      OP_ST: begin
        s.dw = 1'b1; s.d[0] = r32(f.wr[0]); t0 = f.rd[0];
      end
      OP_DSTEP, OP_STREAM: begin   // (DSTEP's wr2 is empty)
        s.dw = 1'b1; s.d[0] = r32(f.wr[0]); s.d[1] = r32(f.wr[2]);
        t0 = f.wr[1]; w0 = 1'b1; t1 = f.rd[0]; t2 = f.rd[1]; t3 = f.rd[2]; t4 = f.rd[3];
      end
      OP_MM: begin
        s.d[0] = r32(f.rd[0]); s.d[1] = r32(f.rd[1]);
        t0 = f.wr[0]; w0 = 1'b1; h = f.wr[1]; t2 = f.rd[3];
        a = f.rd[2];
      end
      OP_QACT: begin
        t0 = f.rd[0]; t1 = f.rd[1]; t2 = f.rd[2];
        s.aw = 1'b1; a = f.wr[0];
      end
      OP_QST: begin
        s.dw = 1'b1; s.d[0] = r32(f.wr[0]); s.d[1] = r32(f.wr[1]);
        t0 = f.rd[0];
      end
      OP_VOP: begin
        t0 = f.wr[0]; w0 = 1'b1; t1 = f.rd[0]; t2 = f.rd[1]; t3 = f.rd[2];
      end
      OP_GATHER: begin
        t0 = f.wr[0]; w0 = 1'b1; t1 = f.rd[0];
      end
      OP_RLD: t0 = f.rd[0];
      OP_WAITW: begin
        s.dw = 1'b1; s.d[0] = r32(f.wr[0]); t0 = f.wr[1]; w0 = 1'b1;
      end
      default: ;   // BAR: all
    endcase
    s.all = f.all || tovf(t0) || tovf(t1) || tovf(t2) || tovf(t3) || tovf(t4) || tovf(h);
    s.t0.w = w0; s.t0.r = hull(rt(t0), rt(h));
    s.t1 = rt(t1);
    s.t2 = rt(t2);
    s.t3 = rt(t3);
    s.t4 = rt(t4);
    s.alo = a.lo[7:0]; s.ahi = a.v ? a.hi[16:0] : '0;
    return s;
  endfunction

  function automatic logic ovr(input r32_t a, input r32_t b);
    return a.lo < b.hi && b.lo < a.hi;
  endfunction

  // ovr() on narrow TMEM ranges
  function automatic logic ovr_t(input rt_t a, input rt_t b);
    return {1'b0, a.lo} < b.hi && {1'b0, b.lo} < a.hi;
  endfunction

  // {conflict, conflict_dram} (otpu_pkg) on segregated footprints: the same-space pairs with at
  // least one write -- 4 DRAM, 9 TMEM (t0 x t0; t0 x t1..t4 both ways) and 1 ACT range pair
  function automatic logic [1:0] conf_s(input fps_t n, input fps_t e);
    logic any, dram;
    any = n.all || e.all; dram = any;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        if ((n.dw || e.dw) && ovr(n.d[i], e.d[j])) begin any = 1'b1; dram = 1'b1; end
    if ((n.t0.w || e.t0.w) && ovr_t(n.t0.r, e.t0.r)) any = 1'b1;
    if (n.t0.w && (ovr_t(n.t0.r, e.t1) || ovr_t(n.t0.r, e.t2) || ovr_t(n.t0.r, e.t3) ||
                   ovr_t(n.t0.r, e.t4))) any = 1'b1;
    if (e.t0.w && (ovr_t(n.t1, e.t0.r) || ovr_t(n.t2, e.t0.r) || ovr_t(n.t3, e.t0.r) ||
                   ovr_t(n.t4, e.t0.r))) any = 1'b1;
    if ((n.aw || e.aw) && {9'd0, n.alo} < e.ahi && {9'd0, e.alo} < n.ahi) any = 1'b1;
    return {any, dram};
  endfunction

`ifndef SYNTHESIS
  // ---- EXPERIMENTAL (docs/seq_vpu.md): emulations of cheaper hazard checks, simulation only.
  // Plusargs (the defaults are today's check, and change nothing):
  //   +hz_tg=G / +hz_dg=G  TMEM / DRAM ranges widened to 2^G-word / -byte granules (lo down, hi
  //                        up): a check on the ranges' top bits only, or region busy bits
  //   +hz_rh=1             an instruction's TMEM reads as one range (their hull): 3 TMEM range
  //                        pairs to compare instead of 9;  +hz_dh=1: its DRAM ranges as one
  //   +hz_win=N            N window slots used;  +hz_uq=a:b:c:d:e: at most that many slots per
  //                        unit (DMA, MXU, QUANT, VPU, COLL; 0: no cap): per-unit queues
  //   +hz_ino=1            every unit starts its instructions in program order (as the MXU and the
  //                        collective unit do today);  +hz_inom=M: only the units of mask M (bit u:
  //                        unit u; 1 DMA, 4 QUANT, 8 VPU)
  //   +hz_rob=1            window slots are freed in program order (a reorder buffer)
  //   +hz_wc=FILE          with +hz_ino: the dependencies from per-pc wait counts (FILE, written by
  //                        +hz_kout) instead of the ranges: wait until at most k_u - 1 of the
  //                        instruction's older unit-u instructions are outstanding
  //   +hz_kout=FILE        write the wait counts: per pc and unit, the least rank (1 = the unit's
  //                        last instruction) of an older instruction it conflicts with, over the
  //                        pc's dynamic instances (the compiler's static view, no loop peeling)
  //   +hz_wcvx=X           with +hz_wc: a VPU producer's dependencies exact (the ranges) instead
  //                        of its count: X = 1 for every consumer, 2 for a VPU consumer only (the
  //                        VPU<-VPU case), 3 for the other units' consumers only
  //   +hz_kclamp=K         with +hz_wc: each count clamped to K (fewer bits a field: K = 1 is a
  //                        bit per unit, "wait for all its older instructions")
  //   +hz_kc=1             the oracle on the window's (coarsened) footprints: a compiler that
  //                        knows the ranges only to the granules of +hz_tg / +hz_dg
  //   +hz_log=FILE         hazard statistics at the end of the run (stall cycles by unit and by
  //                        what they wait for, dependencies by class and kind, rank histograms)
  int  hz_tg = 0, hz_dg = 0, hz_win = 0, hz_kclamp = 0, hz_inom = 0, hz_wcvx = 0;
  bit  hz_rh = 0, hz_dh = 0, hz_ino = 0, hz_rob = 0, hz_wc = 0, hz_cs = 0, hz_kc = 0;
  int  hz_uq [NUNITS];
  initial begin
    string s;
    void'($value$plusargs("hz_tg=%d", hz_tg));
    void'($value$plusargs("hz_dg=%d", hz_dg));
    void'($value$plusargs("hz_win=%d", hz_win));
    void'($value$plusargs("hz_kclamp=%d", hz_kclamp));
    void'($value$plusargs("hz_wcvx=%d", hz_wcvx));
    hz_kc = $test$plusargs("hz_kc=1");
    hz_rh = $test$plusargs("hz_rh=1");
    hz_dh = $test$plusargs("hz_dh=1");
    hz_ino = $test$plusargs("hz_ino=1");
    void'($value$plusargs("hz_inom=%d", hz_inom));
    if (hz_inom != 0) hz_ino = 1'b1;
    else if (hz_ino) hz_inom = (1 << U_DMA) | (1 << U_Q) | (1 << U_VPU);
    hz_rob = $test$plusargs("hz_rob=1");
    for (int u = 0; u < NUNITS; u++) hz_uq[u] = 0;
    if ($value$plusargs("hz_uq=%s", s))
      void'($sscanf(s, "%d:%d:%d:%d:%d", hz_uq[0], hz_uq[1], hz_uq[2], hz_uq[3], hz_uq[4]));
    hz_cs = hz_tg > 0 || hz_dg > 0 || hz_rh || hz_dh;
  end
  fps_t           c_fpx;                    // C's footprint as fp_seg makes it (the oracle's)
  logic [WIN-1:0] sdone;                    // +hz_rob: completed, the slot not yet freed
  logic [WIN-1:0] wc_dep, wc_depd;          // +hz_wc: the dependencies the wait counts give

  // a TMEM range out to 2^g-word granules (hi stays within TAW+1 bits: TMEM ends on a granule)
  function automatic rt_t hz_gt(input rt_t r, input int g);
    rt_t o;
    logic [TAW+1:0] m, h;
    o = r;
    if (r.hi != '0 && g > 0) begin
      m = (g >= TAW) ? {1'b0, {(TAW + 1){1'b1}}} : (TAW + 2)'((64'd1 << g) - 1);
      h = ({1'b0, r.hi} + m) & ~m;
      o.lo = r.lo & ~TAW'(m);
      o.hi = h[TAW+1] ? {1'b1, {TAW{1'b0}}} : h[TAW:0];
    end
    return o;
  endfunction

  function automatic r32_t hz_gd(input r32_t r, input int g);
    r32_t o;
    logic [63:0] m, h;
    o = r;
    if (r.hi != '0 && g > 0) begin
      m = (64'd1 << g) - 1;
      h = (64'(r.hi) + m) & ~m;
      o.lo = r.lo & ~DAW'(m);
      o.hi = ((h >> DAW) != 0) ? '1 : DAW'(h);
    end
    return o;
  endfunction

  function automatic r32_t hz_hull32(input r32_t a, input r32_t b);
    r32_t o;
    if (b.hi == '0) o = a;
    else if (a.hi == '0) o = b;
    else begin
      o.lo = (b.lo < a.lo) ? b.lo : a.lo;
      o.hi = (b.hi > a.hi) ? b.hi : a.hi;
    end
    return o;
  endfunction
`endif

  // the footprint as the window keeps it: fp_seg's, or (simulation) a cheaper check's superset
  function automatic fps_t hz_coarse(input fps_t s);
    fps_t o;
    o = s;
`ifndef SYNTHESIS
    if (hz_rh) begin
      rt_t h;
      h = hull(hull(s.t1, s.t2), hull(s.t3, s.t4));
      if (!s.t0.w) begin h = hull(h, s.t0.r); o.t0.r = '0; end
      o.t1 = h; o.t2 = '0; o.t3 = '0; o.t4 = '0;
    end
    if (hz_dh) begin o.d[0] = hz_hull32(s.d[0], s.d[1]); o.d[1] = '0; end
    if (hz_tg > 0) begin
      o.t0.r = hz_gt(o.t0.r, hz_tg); o.t1 = hz_gt(o.t1, hz_tg); o.t2 = hz_gt(o.t2, hz_tg);
      o.t3 = hz_gt(o.t3, hz_tg); o.t4 = hz_gt(o.t4, hz_tg);
    end
    if (hz_dg > 0) begin o.d[0] = hz_gd(o.d[0], hz_dg); o.d[1] = hz_gd(o.d[1], hz_dg); end
`endif
    return o;
  endfunction

  // window
  logic [WIN-1:0]  sv;                       // slot valid (dispatched, not completed)
  logic [WIN-1:0]  older [WIN];              // older[i][j]: slot j was dispatched before slot i
  logic [NUNITS-1:0] soh [WIN];              // the slot's unit, one-hot
  logic [WIN-1:0]  sstarted, sready;
  logic [WIN-1:0]  sdep [WIN];               // sdep[i][j]: slot i waits for slot j
  logic [WIN-1:0]  sdepd [WIN];              // the DRAM part of sdep (MXU stream start)
  logic [WIN-1:0]  srel;                     // MXU: released
  logic [SW:0]     uq_r;                     // MXU: next started slot to release
  fps_t            sfp  [WIN];
  logic [31:0]     srdy_c [WIN];
  // per-unit queues of started slot ids, in start (= completion) order
  logic [SW-1:0]   uq [NUNITS][WIN];
  logic [SW:0]     uq_h [NUNITS], uq_t [NUNITS];

  // ---- fetch: the row holding fa_d is read at the clock edge that makes it fa; ir takes the
  // instruction at fa at the edge that makes fa pc
  // The RAM reads fa_n on the cycles fa moves (fe) and holds fa's row otherwise: the decision
  // whether R moves on (the dispatch, from the window) reaches the read enable, not the
  // address multiplexer (the sequencer's worst path at 125 MHz ran window -> ld -> address)
  // IMEM: a technology RAM (rtl/tech), one row write (the loader) and one row read per cycle
  otpu_ram_1r1w #(.W(D*8), .N(NROW), .STYLE("auto"), .ZERO(1'b0)) u_imem (
    .clk, .we(im_we), .wa(RW'(im_row)), .wd(im_data), .re(fe), .ra(RW'(fa_n / IPR)), .rd(irow));
  // LI / ADDI and rd, decoded as ir loads: R's write enables start at flip-flops (R's process)
  logic        ir_rw;
  logic [15:0] ir_rdh;
  always_ff @(posedge clk) begin
    fa <= fa_d;
    if (rst) ir_v <= 1'b0;
    else if (ld) begin
      ir <= irow[256 * (fa % IPR) +: 256];
      ir_v <= 1'b1;
      ir_rw <= irow[256 * (fa % IPR) +: 8] == OP_LI || irow[256 * (fa % IPR) +: 8] == OP_ADDI;
      ir_rdh <= (16'd1 << irow[256 * (fa % IPR) + 20 +: 4]) & ~16'd1;
    end
  end
  // WAITC: the sidecar RAM of wait entries, read beside IMEM: rows of 4 entries of SWW bits
  // (the counts' 28, the VPU mask's VMW; 1024 x 236 bits for a 4096-instruction IMEM). The
  // loader writes a row from 4 entries as they are in DRAM (64 bits each: the counts' word,
  // then the mask's)
  localparam int VMW = 31;        // the VPU mask: the VPU instructions 1..VMW back
  localparam int SWW = 28 + VMW;
  localparam int NSROW = IMEM_WORDS / 32;
  localparam int SRW = $clog2(NSROW);
  logic [SWW-1:0] ir_sync;        // the wait entry of the instruction at pc
  if (WAITC) begin : g_sync
    logic [4*SWW-1:0] srow, sc_pk;
    logic         sc_off;         // simulation: +hz_sync's entries stay (the loader's are dropped)
`ifdef SYNTHESIS
    assign sc_off = 1'b0;
`endif
    always_comb
      for (int e = 0; e < 4; e++)
        sc_pk[SWW * e +: SWW] = {sc_data[64 * e + 32 +: VMW], sc_data[64 * e +: 28]};
    otpu_ram_1r1w #(.W(4 * SWW), .N(NSROW), .STYLE("auto"), .ZERO(1'b0)) u_sync (
      .clk, .we(sc_we && !sc_off), .wa(SRW'(sc_row)), .wd(sc_pk), .re(fe),
      .ra(SRW'(fa_n / 4)), .rd(srow));
    always_ff @(posedge clk) if (!rst && ld) ir_sync <= srow[SWW * (fa % 4) +: SWW];
`ifndef SYNTHESIS
    // simulation: the wait entries from +hz_sync_s<SID>=FILE or +hz_sync=FILE (lines "pc k0 ..
    // k6 n [mask]": opentpu.waits' with the VPU mask; +hz_kout's, the oracle's, without: the
    // mask is then the VPU instructions k3 .. VMW back), each count clamped to +hz_kclamp; an
    // instruction the file does not name waits for every unit's last instruction and the
    // VPU's VMW. With a file the loader's entries (boot, HALT CHAIN) are dropped
    initial begin
      string fn, ln;
      int fd, pc, r, n, kc, mk;
      int k [7];
      kc = 0;
      fn = "";
      void'($value$plusargs("hz_kclamp=%d", kc));
      for (int rr = 0; rr < NSROW; rr++)
        for (int w = 0; w < 4; w++) u_sync.g_auto.mem[rr][SWW * w +: SWW] = {{VMW{1'b1}}, 28'h111_1111};
      if (!$value$plusargs($sformatf("hz_sync_s%0d=%%s", SID), fn))
        void'($value$plusargs("hz_sync=%s", fn));
      sc_off = (fn != "");
      if (fn != "") begin
        fd = $fopen(fn, "r");
        if (fd == 0) $fatal(1, "otpu_seq: +hz_sync=%s: cannot open", fn);
        while ($fgets(ln, fd) != 0) begin
          mk = -1;
          r = $sscanf(ln, "%d %d %d %d %d %d %d %d %d %d", pc, k[0], k[1], k[2], k[3], k[4],
                      k[5], k[6], n, mk);
          if (r >= 8 && pc >= 0 && pc < IMEM_WORDS / 8) begin
            logic [SWW-1:0] wd;
            wd = '0;
            for (int j = 0; j < 7; j++)
              wd[4 * j +: 4] = 4'((kc > 0 && k[j] > kc) ? kc : (k[j] > 15) ? 15 : k[j]);
            if (r >= 10) wd[28 +: VMW] = VMW'(mk);
            else if (k[3] > 0) wd[28 +: VMW] = {VMW{1'b1}} << ((k[3] > 15 ? 15 : k[3]) - 1);
            u_sync.g_auto.mem[pc / 4][SWW * (pc % 4) +: SWW] = wd;
          end
        end
        $fclose(fd);
      end
    end
`endif
  end else begin : g_nosync
    assign ir_sync = '0;
  end
  logic [31:0] iw [8];
  always_comb for (int k = 0; k < 8; k++) iw[k] = ir[32 * k +: 32];
  wire [7:0] op    = iw[0][7:0];
  wire [3:0] ra    = iw[0][11:8];
  wire [3:0] rb    = iw[0][15:12];
  wire [3:0] rc    = iw[0][19:16];
  wire [3:0] rd    = iw[0][23:20];
  wire [7:0] flags = iw[0][31:24];

  function automatic logic [31:0] rv(input logic [3:0] r);
    return (r == 0) ? 32'd0 : R[r];
  endfunction

  // STREAM: src (w2) and dst (w3) += R[ra], vec (w4) += R[rb], x (w5) += R[rc], k (w6) += R[rd].
  // The others: w1 += R[ra], w2 += R[rb] and w3 += R[rc] where docs/isa.md writes the field as
  // R[x] + w; the counts and shapes there are immediates: QACT's w2 (rows, ab, KB) and w3 (srs),
  // LD's and ST's w3 (n), GATHER's w3 (rows, cols) (the enables decode the opcode beside the
  // register reads)
  cmd_t dcmd;
  wire  is_str = (op == OP_STREAM);
  wire  rel2 = (op != OP_QACT);
  wire  rel3 = !(op == OP_QACT || op == OP_LD || op == OP_ST || op == OP_GATHER);
  always_comb begin
    dcmd.op = op; dcmd.flags = flags;
    dcmd.w1 = iw[1] + (is_str ? 32'd0 : rv(ra));
    dcmd.w2 = iw[2] + (is_str ? rv(ra) : rel2 ? rv(rb) : 32'd0);
    dcmd.w3 = iw[3] + (is_str ? rv(ra) : rel3 ? rv(rc) : 32'd0);
    dcmd.w4 = iw[4] + (is_str ? rv(rb) : 32'd0);
    dcmd.w5 = iw[5] + (is_str ? rv(rc) : 32'd0);
    dcmd.w6 = iw[6] + (is_str ? rv(rd) : 32'd0);
    dcmd.w7 = iw[7] + ((op == OP_VOP) ? rv(rd) : 32'd0);        // VOP: w7 += R[rd]
    // DAW > 32: the instruction's DRAM region REG from its spare bits (isa.py "DRAM address
    // width"): rd[RGN-1:0] (rd names no register of these ops: only LI, ADDI, RLD, VOP's w7
    // and STREAM's k use it) and STREAM's flags[7:4] (its flags 3:0 are the DMA's). Straight from
    // ir, off the adders' path; at DAW 32 a constant 0 (the bits are ignored)
    dcmd.rg = '0;
    if (WIDE)
      case (op)
        OP_LD, OP_ST, OP_DSTEP, OP_WAITW, OP_MM, OP_QST: dcmd.rg = RGW'(rd);
        OP_STREAM: dcmd.rg = RGW'(flags[7:4]);
        default: ;
      endcase
  end
  int dunit;
  always_comb dunit = (!STREAMS && (op == OP_DSTEP || op == OP_STREAM)) ? -1 : unit_of(op);

  // R's writes: LI / ADDI here, with enables from flip-flops decoded as ir loaded (ir_rw, ir_rdh;
  // 250 MHz, FK35: ir's opcode -> the decode -> R's enables, 4 levels, 87% routing, -0.74 ns),
  // and an RLD's value, which lands while R holds (rl_wait), so never beside an LI / ADDI
  wire        r_wr = !stopping && !halted && ir_v && !rl_wait && ir_rw;
  wire [31:0] r_wd = (op == OP_LI) ? iw[1] : rv(ra) + iw[1];
  always_ff @(posedge clk) begin
    if (rst) begin
      for (int i = 0; i < 8; i++) R[i] <= '0;
      for (int i = 0; i < 8; i++) R[8 + i] <= rinit[i];
    end else
      for (int k = 1; k < 16; k++)
        if (rl_v && rl_rd == 4'(k)) R[k] <= rl_val;
        else if (r_wr && ir_rdh[k]) R[k] <= r_wd;
  end

  // ---- dispatch pipeline: R (the instruction at pc: registers resolved) -> P -> O (the
  // footprint's multiplications' operands) -> S (partial products) -> Q (products) -> F
  // (footprint ranges) -> Y (the window's form, fp_seg) -> a FIFO -> K (dependencies on the window,
  // registered) -> C (slot allocation). In order. P .. Y never stall (no enable on their
  // registers): R hands an instruction to P only while the stages and the FIFO together hold
  // fewer than NF (credits, cnt), so the FIFO always has room for what they carry; K and C hold
  // while the window is full. The stages stream, so they add latency only at the program start
  // and after a full window. (250 MHz, FK35: the stalls rippled from the window through C, Q, S
  // and P into R's ld, ir's enable and IMEM's read enable, sv -> ir CE, 6 levels, -0.79 ns; and
  // Q's ranges and fp_seg in one cycle, q_cmd -> c_fp, 17-19 levels, -0.82 ns.) WAITC: P -> C,
  // no footprint stages, as before.
  localparam int NF = 8;                        // the FIFO's entries: R's credits
  logic        p_v, o_v, s_v, q_v, f_v, y_v, k_v, c_v;
  cmd_t        p_cmd, o_cmd, s_cmd, q_cmd, f_cmd, y_cmd, k_cmd, c_cmd;
  logic [2:0]  p_unit, o_unit, s_unit, q_unit, f_unit, y_unit, k_unit, c_unit;
  logic [31:0] p_pc, o_pc, s_pc, q_pc, f_pc, y_pc, k_pc, c_pc;
  fpo_t        o_ops;
  fpp_t        s_pp;
  fpm_t        q_pr;
  fp_t         q_fp;             // Q's footprint ranges (otpu_pkg form)
  fp_t         f_fp;
  fps_t        y_fp, k_fp, c_fp;
  logic [3:0]  cnt;              // instructions in O .. Y and the FIFO (P's is p_v)
  logic [3:0]  fw, fr;           // the FIFO's write and read pointers (a wrap bit above 3)
  wire         f_ne = fw != fr;
  logic        have_free;
  logic        wc_hold;          // WAITC: a VPU instruction waits for the one VMW back
  wire         c_go = c_v && have_free && !wc_hold;
  wire         kc = !c_v || c_go;               // K hands its instruction to C
  wire         k_pop = !WAITC && f_ne && (!k_v || kc);   // K takes the FIFO's head
  wire         p_adv = WAITC ? (!c_v || c_go) : 1'b1;    // P hands its instruction on
  logic [SWW-1:0] p_sync, c_sync;               // WAITC: the instruction's wait entry
  // P can take the instruction at R (credits: P's own and those in O .. the FIFO below NF)
  wire         r_take = WAITC ? (!p_v || p_adv) : (p_v ? cnt < 4'(NF - 1) : cnt < 4'(NF));
  wire         pipe_empty = WAITC ? (!p_v && !c_v) : (!p_v && cnt == '0 && !k_v && !c_v);
  // K's dependencies on the window and on C's instruction, registered as K hands to C: C's
  // slot is the free slot of that cycle if it dispatched then (c_xg), and a slot of the window
  // whose instruction completed since has sv low. Nothing else enters the window meanwhile (C is
  // its only writer), so they stay exact while C waits for a free slot.
  logic [1:0]  c_cf [WIN];
  logic [1:0]  c_cx;
  logic [SW-1:0] c_xs;
  logic        c_xg;
  // R retires a control instruction this cycle (NOP, HALT, LI, ADDI, a LOOP's second cycle)
  logic        r_ret;

  // ---- completions: the slots whose units reported done a cycle ago (registered: 250 MHz, FK35:
  // uq_h -> the unit queue's read -> fin -> sdep, 6 levels, -0.56 ns). A slot is freed, and its
  // dependents' bits cleared, a cycle after its udone; both in the same cycle, as before.
  logic [WIN-1:0] fin, fin_c;
  always_comb begin
    fin_c = '0;
    for (int u = 0; u < NUNITS; u++)
      if (udone[u]) fin_c[uq[u][uq_h[u][SW-1:0]]] = 1'b1;
  end
  always_ff @(posedge clk) fin <= rst ? '0 : fin_c;

  // ---- free slot and the new instruction's dependencies
  logic [SW-1:0]   free_slot;
  logic [WIN-1:0]  ndep, ndepd;
  always_comb q_fp = fp_ranges(q_cmd, q_pr, D, S);

  // P .. Y's data: no enable, no reset (their valid bits are in the main process)
  always_ff @(posedge clk) begin
    o_cmd <= p_cmd; o_unit <= p_unit; o_pc <= p_pc; o_ops <= fp_ops(p_cmd, D, S);
    s_cmd <= o_cmd; s_unit <= o_unit; s_pc <= o_pc; s_pp <= fp_part(o_ops);
    q_cmd <= s_cmd; q_unit <= s_unit; q_pc <= s_pc; q_pr <= fp_sum(s_pp);
    f_cmd <= q_cmd; f_unit <= q_unit; f_pc <= q_pc; f_fp <= q_fp;
    y_cmd <= f_cmd; y_unit <= f_unit; y_pc <= f_pc; y_fp <= hz_coarse(fp_seg(f_cmd.op, f_fp));
  end
  // the FIFO between Y and K: distributed RAM (one write, one asynchronous read at fr, a flip-flop);
  // R's credits keep it from overflowing
  typedef struct packed { cmd_t cmd; logic [2:0] unit; logic [31:0] pc; fps_t fp; } fqe_t;
  (* ram_style = "distributed" *) fqe_t fq [NF];
  localparam int FW = $clog2(NF);
  always_ff @(posedge clk) if (!WAITC && y_v) fq[fw[FW-1:0]] <= {y_cmd, y_unit, y_pc, y_fp};
  always_ff @(posedge clk) begin
    if (rst) begin
      fw <= '0; fr <= '0;
    end else begin
      if (!WAITC && y_v) fw <= fw + 1'b1;
      if (k_pop) fr <= fr + 1'b1;
    end
    if (k_pop) begin
      k_cmd <= fq[fr[FW-1:0]].cmd; k_unit <= fq[fr[FW-1:0]].unit;
      k_pc <= fq[fr[FW-1:0]].pc; k_fp <= fq[fr[FW-1:0]].fp;
    end
`ifndef SYNTHESIS
    if (!rst && !WAITC && y_v && 4'(fw - fr) == 4'(NF))
      $fatal(1, "otpu_seq: the dispatch FIFO overflows (credits)");
`endif
  end
  always_comb begin
    have_free = 1'b0; free_slot = '0;
    for (int i = WIN - 1; i >= 0; i--)
      if (!sv[i]) begin have_free = 1'b1; free_slot = SW'(i); end
    // (K's conf_s, registered: c_cf, c_cx; 250 MHz, FK35: c_fp -> conf_s -> ndep -> sdep, 7 levels,
    // -0.56 ns)
    for (int i = 0; i < WIN; i++) begin
      logic [1:0] cf;
      cf = (c_xg && c_xs == SW'(i)) ? c_cx : c_cf[i];
      ndep[i] = sv[i] && !fin[i] && cf[1];
      ndepd[i] = sv[i] && !fin[i] && cf[0];
    end
`ifndef SYNTHESIS
    // the emulations: window / per-unit caps, the reorder buffer's held slots, wait counts
    begin
      int n, nu;
      n = 0; nu = 0;
      for (int i = 0; i < WIN; i++) begin
        n += int'(sv[i]);
        nu += int'(sv[i] && c_unit < NUNITS && soh[i][c_unit]);
      end
      if ((hz_win > 0 && n >= hz_win) ||
          (c_unit < NUNITS && hz_uq[c_unit] > 0 && nu >= hz_uq[c_unit]))
        have_free = 1'b0;
      if (hz_rob) begin ndep &= ~sdone; ndepd &= ~sdone; end
      if (hz_wc) begin ndep = wc_dep; ndepd = wc_depd; end
    end
`endif
    if (WAITC) begin ndep = '0; ndepd = '0; end   // the wait counts instead (w_all, w_dram)
  end

  // ---- WAITC: per unit, its instructions dispatched (wd_n) and, the in-order units, completed
  // (wc_n: each completes its instructions in program order); per slot the counts its waits
  // need: wait j (j < NUNITS: unit j; 5, 6: the DMA's and the quantizer's for an MM's stream
  // start) holds until wc_n of its unit reaches wt (its producer's index + 1; the counters wrap,
  // so the test is on the sign of the difference: a slot's target stays within WIN of the
  // count). The unit's own wait is ignored: the unit starts in order. The VPU starts its oldest
  // ready instruction and completes them out of order: a slot's VPU producers are a mask over
  // the VPU instructions outstanding (vout, by index mod VMW + 1), cleared as each completes. A
  // VPU instruction is dispatched once the one VMW before it has completed (wc_hold), so every
  // older one has (each younger one's dispatch waited for its own), and a mask of VMW names
  // every one a later instruction may wait for. A VPU instruction's slot is freed when it
  // completes, so younger ones pass an old one blocked on its producers: the hold binds when
  // VMW of them have (docs/seq_vpu.md 4.5: VMW 15 cost up to 0.8% of prefill; 31 nothing on
  // the runs measured)
  localparam int CW = 8;
  localparam int NWT = NUNITS + 2;
  logic [CW-1:0]    wd_n [NUNITS], wc_n [NUNITS];
  localparam int VN = VMW + 1;                // VPU indices kept (mod VN)
  localparam int VIW = $clog2(VN);
  logic [VN-1:0]    vout;                     // VPU instructions outstanding (index mod VN)
  logic [VN-1:0]    vpend [WIN];              // a slot's VPU producers still outstanding
  logic [VIW-1:0]   vix [WIN];                // a VPU slot's index mod VN
  logic [VN-1:0]    v_fin;                    // the VPU's completion this cycle (index mod VN)
  // vout at the next VPU instruction's index VMW back (wd_n + 1 mod VN), a flip-flop loaded with
  // its next value (vout's, at wd_n's next): wc_hold, so the dispatch (c_go -> P -> R's advance
  // -> the loop counters' enables, pc) starts at it, not at a 32:1 select of vout by wd_n (the m8
  // card at 150 MHz: vout -> wc_hold -> c_go -> stk_rem's enables, 9 levels, -0.063 ns)
  logic             vnx_q;
  logic [CW-1:0]    wt [WIN][NWT];
  logic [NWT-1:0]   wv [WIN];
  logic [WIN-1:0]   w_all, w_dram;            // the slot's waits (all / the stream's) are met
  // ... registered (release r2): what the starts, the MXU's release and the ready stamp test.
  // The counts' compares (wc_n - wt per slot and wait) end at these flip-flops instead of
  // running on through the start selection into every unit's command register (r2m4, 133.33
  // MHz: wc_n -> g_cmd q, 358 endpoints, -0.482 ns, 11.5 levels, 16% of core_clk's TNS). A wait
  // is released a cycle after its count is reached. A slot's readiness only rises while it holds
  // its instruction (the counts and the VPU mask only move towards its targets), so last cycle's
  // is never wrong for this one, except for the slot written this cycle: its flag clears, and
  // an instruction starts two cycles after its dispatch at the earliest
  logic [WIN-1:0]   w_all_q, w_dram_q;
`ifndef SYNTHESIS
  // EXPERIMENTAL, simulation only (docs/seq_vpu.md): +wc_vx=1 takes a slot's VPU producers from
  // today's range check (fp_seg, conf_s: vxd) instead of its mask; +wc_vgate=G holds a VOP
  // until the one G back has completed instead of VMW (G = 0: never; G > VMW or 0 only with
  // +wc_vx, the mask names VMW)
  bit             wc_vx = 1'b0;
  int             wc_vg = VMW;
  logic [WIN-1:0] vxd [WIN];
  fps_t           sfpx [WIN];
  logic [CW-1:0]  vxi [WIN];
  initial begin
    wc_vx = $test$plusargs("wc_vx=1");
    void'($value$plusargs("wc_vgate=%d", wc_vg));
    if ((wc_vg == 0 || wc_vg > VMW) && !wc_vx)
      $fatal(1, "otpu_seq: +wc_vgate=%0d needs +wc_vx=1", wc_vg);
  end
`endif
  function automatic int wunit(input int j);
    return (j < NUNITS) ? j : (j == NUNITS) ? U_DMA : U_Q;
  endfunction
  always_comb begin
    v_fin = '0;
    if (udone[U_VPU]) v_fin[vix[uq[U_VPU][uq_h[U_VPU][SW-1:0]]]] = 1'b1;
    wc_hold = WAITC && c_unit == U_VPU && vnx_q;
`ifndef SYNTHESIS
    if (wc_vg != VMW) begin
      wc_hold = 1'b0;
      if (WAITC && c_unit == U_VPU && wc_vg > 0)
        for (int j = 0; j < WIN; j++)
          if (sv[j] && soh[j][U_VPU] && CW'(wd_n[U_VPU] - vxi[j]) >= CW'(wc_vg)) wc_hold = 1'b1;
    end
`endif
  end
`ifndef SYNTHESIS
  bit vnx_seen;                                // a reset seen (registers start arbitrary)
  initial vnx_seen = 1'b0;
  always @(posedge clk) if (rst) vnx_seen <= 1'b1;
  always @(posedge clk)
    if (WAITC && !rst && vnx_seen && vnx_q != vout[VIW'(wd_n[U_VPU] + 1'b1)])
      $fatal(1, "otpu_seq: vnx_q %0d is not vout[%0d]", vnx_q, VIW'(wd_n[U_VPU] + 1'b1));
`endif
  always_comb
    for (int i = 0; i < WIN; i++) begin
      logic [NWT-1:0] ok;
      for (int j = 0; j < NWT; j++) begin
        logic [CW-1:0] df;
        df = wc_n[wunit(j)] - wt[i][j];
        ok[j] = !wv[i][j] || !df[CW-1];
      end
      w_all[i] = &ok[NUNITS-1:0] && vpend[i] == '0;
`ifndef SYNTHESIS
      if (wc_vx) w_all[i] = &ok[NUNITS-1:0] && vxd[i] == '0;
`endif
      w_dram[i] = ok[NUNITS] && ok[NUNITS+1];
    end
  always_ff @(posedge clk)
    if (rst) begin
      w_all_q <= '0;
      w_dram_q <= '0;
    end else begin
      w_all_q <= w_all & ~(c_go ? WIN'(1) << free_slot : '0);
      w_dram_q <= w_dram & ~(c_go ? WIN'(1) << free_slot : '0);
    end

  // ---- loop-end handling for the instruction at pc: at_end_q is (sp != 0) && stk_end[sp-1] ==
  // pc, registered as R moves on (with the stack as it leaves the move; below)
  logic        at_end;
  always_comb at_end = at_end_q;

  task automatic advance();
    if (at_end) begin
      if (t_gt1) begin
        stk_rem[sp-1] <= t_rem - 1;
        t_rem <= t_rem - 1; t_gt1 <= t_gt2; t_gt2 <= t_rem > 32'd3;
      end else begin
        sp <= sp - 1;
        t_start <= n_start; t_end <= n_end; t_rem <= n_rem; t_gt1 <= n_gt1; t_gt2 <= n_rem > 32'd2;
        n_start <= stk_start[sp-3]; n_end <= stk_end[sp-3]; n_rem <= stk_rem[sp-3];
        n_gt1 <= stk_rem[sp-3] > 32'd1;
      end
    end
  endtask

  always_comb
    r_ret = ir_v && !stopping && !halted && !rl_wait &&
            (op == OP_NOP || op == OP_HALT || op == OP_LI || op == OP_ADDI ||
             (op == OP_LOOP && lp == 2'd2));

  // ---- the next fetch address (mirrors the fetch/dispatch below). fa is the successor of pc
  // under the loop stack as it stands while R holds pc; when R moves on, fa_d is the successor
  // of fa under the stack as R's instruction leaves it:
  //   A: a LOOP's third cycle pushes (fa == pc + 1 then) -- a one-instruction body repeats
  //   B: advance() at a loop end decrements the top      -- repeats if the top keeps rem > 1
  //   C: advance() at a loop end pops                    -- the next entry decides
  //   D: the stack is unchanged (also the load after reset and a skipped LOOP)
  // A LOOP's second cycle fetches the enter/skip target a cycle ahead (pc holds, the RAM reads
  // it). The stack's top two entries and their flags are flip-flops (t_*, n_*).
  logic        adv;              // R's instruction retires or is handed over this cycle
  logic        hit1, hit2;       // fa ends the loop on top of the stack / the one below it
  logic        hit;              // fa's successor is a loop start ...
  logic [31:0] hit_pc;           // ... this one
  logic [31:0] succ;
  always_comb begin
    case (op)
      OP_NOP, OP_LI, OP_ADDI: adv = 1'b1;
      OP_HALT: adv = 1'b0;
      OP_LOOP: adv = lp == 2'd2;
      default: adv = dunit >= 0 && r_take;
    endcase
    ld = !ir_v || (!stopping && !halted && !rl_wait && adv);
    hit1 = (sp != 0) && (t_end == fa);
    hit2 = (sp >= 2) && (n_end == fa);
    hit = hit1 && t_gt1;                                                      // D
    hit_pc = t_start;
    if (ir_v && op == OP_LOOP) begin
      if (lp_nz) begin hit = lp_len1 && lp_gt1; hit_pc = fa; end              // A
    end else if (ir_v && at_end) begin
      if (t_gt1) hit = hit1 && t_gt2;                                         // B
      else begin hit = hit2 && n_gt1; hit_pc = n_start; end                   // C
    end
    succ = hit ? hit_pc : fa + 1;
    if (rst) fa_n = '0;
    else if (ir_v && op == OP_LOOP && lp == 2'd1) fa_n = (lp_cnt == '0) ? lp_skip : pc + 1;
    else fa_n = succ;
    fe = rst || (ir_v && op == OP_LOOP && lp == 2'd1 && !stopping && !halted && !rl_wait) || ld;
    fa_d = fe ? fa_n : fa;
  end

  // A dependency on an instruction that has already started on the SAME unit is satisfied:
  // every unit processes its instructions in start order (the MXU consumes and drains its two
  // commands strictly in order), so the older one's effects land first. This lets the MXU
  // stream an accumulating MM while the previous one into the same tile is still finishing.
  // The VPU is the exception: it overlaps several instructions in its pipeline (with different
  // latencies), so a VPU instruction waits for the VPU instructions it depends on to complete.
  logic [WIN-1:0] same_started [NUNITS];
  always_comb begin
    for (int u = 0; u < NUNITS; u++)
      for (int i = 0; i < WIN; i++)
        same_started[u][i] = sv[i] && sstarted[i] && soh[i][u] && u != U_VPU;
  end

  // ---- per-unit start: the oldest ready (all dependencies completed) instruction
  logic [NUNITS-1:0] can_start;
  logic [SW-1:0]     start_slot [NUNITS];
  logic [WIN-1:0]    sel [NUNITS];           // the oldest candidate, one-hot (or none)
  logic [NUNITS-1:0] cand_any;               // the unit has a candidate
  logic [WIN-1:0]    mrdy, crdy;             // per slot: ready as the MXU / COLL oldest
  // (the oldest of a set: the member none of whose older slots is in the set -- an age matrix
  // instead of comparing dispatch numbers, so the choice is parallel logic). older is a strict
  // total order over the valid slots, so sel has at most one bit; readiness is tested per slot
  // in parallel with the selection and AND-ORed with sel -- no slot id on the can_start path.
  // The choice is made from flip-flops (250 MHz, FK35: sv / sstarted -> same_started -> every
  // slot's readiness -> the age matrix -> start_slot -> the command store's read -> g_cmd q,
  // 8 levels, -0.95 ns at 4 ns): a slot's readiness as each unit sees it (rdy_q: the MXU's the
  // stream start's, sdepd; rrdy_q: the MXU release's, sdep), then each unit's candidates (cand_q)
  // and, per candidate, whether an older candidate exists (blk_q) are registered, and the start
  // only masks out what started meanwhile (sstarted). Neither is ever wrong a cycle late:
  //  - readiness only rises while a slot holds its instruction (sdep / sdepd bits clear as their
  //    producers complete; a producer that started on the same unit counts as clear and stays
  //    so until it completes, which clears the bit in the same cycle; a later dispatch never
  //    adds a dependency to an older slot). A free slot's is 0 (sv), so a slot written this
  //    cycle starts with its readiness clear, and its own is registered a cycle after.
  //  - cand_q only loses members by starting (masked by sstarted) and gains them a cycle late:
  //    a younger member can only start earlier, which is always safe (the oldest READY one is
  //    a performance choice; a dependent never becomes ready before its producer); the strictly
  //    in-order MXU and collective unit take all their unstarted slots as candidates, and a slot
  //    dispatched later is younger than all of them, so their order holds.
  //  - blk_q[i] counts the candidates of a cycle ago: one that started since (the unit then has
  //    ustart this cycle and cannot start anyway) only delays i by a cycle; one dispatched since
  //    is younger. So sel still has at most one bit: the oldest of cand_q, if not started.
  // An instruction now starts two cycles after its dependencies clear or after its dispatch at
  // the earliest (the MXU and the collective unit one), instead of the same cycle.
  logic [WIN-1:0] rdy_q [NUNITS], rrdy_q, cand_q [NUNITS], blk_q [NUNITS];
  always_ff @(posedge clk) begin
    logic [WIN-1:0] cn;
    for (int u = 0; u < NUNITS; u++) begin
      for (int i = 0; i < WIN; i++) begin
        rdy_q[u][i] <= !rst && sv[i] &&
                       (((u == U_MXU) ? sdepd[i] : sdep[i]) & ~same_started[u]) == '0;
        cn[i] = sv[i] && !sstarted[i] && soh[i][u] &&
                (u == U_COLL || u == U_MXU || (WAITC ? (u != U_VPU || w_all_q[i]) : rdy_q[u][i]));
      end
      cand_q[u] <= rst ? '0 : cn;
      for (int i = 0; i < WIN; i++) blk_q[u][i] <= (older[i] & cn) != '0;
    end
    for (int i = 0; i < WIN; i++)
      rrdy_q[i] <= !rst && sv[i] && (sdep[i] & ~same_started[U_MXU]) == '0;
  end
  always_comb begin
    for (int i = 0; i < WIN; i++) begin
      mrdy[i] = WAITC ? w_dram_q[i] : rdy_q[U_MXU][i];
      crdy[i] = WAITC ? w_all_q[i] : rdy_q[U_COLL][i];
    end
    for (int u = 0; u < NUNITS; u++) begin
      start_slot[u] = '0;
      // collectives pair up across slices, so the collective unit stays strictly in order:
      // it only considers its oldest instruction
      for (int i = 0; i < WIN; i++) begin
        sel[u][i] = cand_q[u][i] && !sstarted[i] && !blk_q[u][i];
        if (sel[u][i]) start_slot[u] |= SW'(i);
      end
      cand_any[u] = |(cand_q[u] & ~sstarted);
      if (u == U_MXU)
        can_start[u] = |(sel[u] & mrdy) && urdy[u] && !ustart[u];
      else if (u == U_COLL)
        can_start[u] = |(sel[u] & crdy) && urdy[u] && !ustart[u];
      else    // cand already requires readiness
        can_start[u] = |sel[u] && urdy[u] && !ustart[u];
    end
    // WAITC: the VPU starts its oldest ready instruction (its waits met), the others their
    // oldest only
    if (WAITC)
      for (int u = 0; u < NUNITS; u++)
        if (u != U_MXU && u != U_COLL) begin
          logic [WIN-1:0] c2;
          start_slot[u] = '0;
          for (int i = 0; i < WIN; i++)
            c2[i] = sv[i] && !sstarted[i] && soh[i][u] && (u != U_VPU || w_all_q[i]);
          for (int i = 0; i < WIN; i++) begin
            sel[u][i] = c2[i] && (older[i] & c2) == '0;
            if (sel[u][i]) start_slot[u] |= SW'(i);
          end
          cand_any[u] = |c2;
          can_start[u] = |(sel[u] & w_all_q) && urdy[u] && !ustart[u];
        end
`ifndef SYNTHESIS
    // +hz_ino: the DMA, the quantizer and the VPU start their oldest instruction only, as the
    // MXU and the collective unit do
    if (hz_ino)
      for (int u = 0; u < NUNITS; u++)
        if (u != U_MXU && u != U_COLL && hz_inom[u]) begin
          logic [WIN-1:0] c2, r2;
          start_slot[u] = '0;
          for (int i = 0; i < WIN; i++) begin
            c2[i] = sv[i] && !sstarted[i] && soh[i][u];
            r2[i] = (sdep[i] & ~same_started[u]) == '0;
          end
          for (int i = 0; i < WIN; i++) begin
            sel[u][i] = c2[i] && (older[i] & c2) == '0;
            if (sel[u][i]) start_slot[u] |= SW'(i);
          end
          cand_any[u] = |c2;
          can_start[u] = |(sel[u] & r2) && urdy[u] && !ustart[u];
        end
`endif
  end


  // MXU release: the oldest started, unreleased MM once all its dependencies are clear (rrdy_q:
  // registered as the start's readiness is; a started slot has been valid for cycles)
  wire [SW-1:0] rel_slot = uq[U_MXU][uq_r[SW-1:0]];
  wire          can_rel  = (uq_r != uq_t[U_MXU]) && (WAITC ? w_all_q[rel_slot] : rrdy_q[rel_slot]);

  // Each unit's command is read from the window as the unit starts, at the slot it starts, into a
  // register (ucmd): a slot's command is written only at dispatch and stays until the slot
  // completes, so the copy is the slot's command for as long as the unit reads it (at ustart; the
  // collective until its udone). The units' start decode then begins at flip-flops instead of at
  // the LUT RAM's read (133.33 MHz, 61e015ff4: ucs -> LUT RAM -> the MXU's KBa -> DSP -> n.total,
  // 5 levels, -0.504 ns; ucs -> the quantizer's busy, -0.417).
  // The command store is one distributed RAM per unit (1 write, 1 asynchronous read, no reset):
  // a single read port each, so it infers as LUT-RAM and trims the fields its unit never reads
  // (as do the registers).
  // Each copy loads on every cycle its unit is ready and not starting (urdy, ustart), which covers
  // its starts: the enable is the unit's rdy and a flip-flop, not the start decode (133.33 MHz,
  // 812bb01: sstarted -> the collective's can_start -> its copy's enable, fanout 120, 7 levels,
  // +0.165 ns; the m8 card at a 150 MHz equivalent, unit 2's, 216 loads, -0.028). The units read
  // their command only at ustart (the DMA, the MXU, the quantizer and the VPU copy it then; the
  // MXU's release reads none), when the copy holds (ustart is up), and the collective only while
  // its request is pending (urdy low). Between starts the copy follows start_slot, unread.
  // WAITC: the store is written a cycle after the dispatch, from flip-flops: the enable and the
  // slot a copy per unit (kept), the command one register. A slot starts two cycles after its
  // dispatch at the earliest (w_all_q, w_dram_q), so its command is in the store by then. The
  // dispatch decision and the free slot then end at a few flip-flops instead of every store's
  // write port (r22's m8 at 150 MHz: sv -> the free slot -> the LUT RAMs' write addresses, 3036
  // pins, -0.49 ns; vout -> c_go -> their write enables, 868, -0.34)
  cmd_t c_cmd_w;
  always_ff @(posedge clk) c_cmd_w <= c_cmd;
  logic [SW-1:0] ucs [NUNITS];
  for (genvar u = 0; u < NUNITS; u++) begin : g_cmd
    (* ram_style = "distributed" *) logic [$bits(cmd_t)-1:0] m [WIN];
    logic [$bits(cmd_t)-1:0] q;
    logic          we;
    logic [SW-1:0] wa;
    wire q_en = urdy[u] && !ustart[u];
    if (WAITC) begin : g_wq
      (* keep_hierarchy = "yes" *)
      otpu_delay #(.W(1 + SW), .N(1)) u_w (.clk, .en(1'b1), .d({!rst && c_go, free_slot}),
                                          .q({we, wa}));
    end else begin : g_wd
      assign we = !rst && c_go;
      assign wa = free_slot;
    end
    always_ff @(posedge clk) if (we) m[wa] <= WAITC ? c_cmd_w : c_cmd;
    always_ff @(posedge clk) if (!rst && q_en) q <= m[start_slot[u]];
    assign ucmd[u] = cmd_t'(q);
`ifndef SYNTHESIS
    // the copy is the started slot's command when the unit reads it (at ustart; the collective
    // while its request is pending); no start reads a slot whose write is still on its way
    always @(posedge clk) begin
      if (!rst && (ustart[u] || (u == U_COLL && !urdy[u])) && sv[ucs[u]] && sstarted[ucs[u]]
          && soh[ucs[u]][u] && q != m[ucs[u]])
        $fatal(1, "otpu_seq: unit %0d's command is not slot %0d's", u, ucs[u]);
      if (!rst && can_start[u] && !q_en)
        $fatal(1, "otpu_seq: unit %0d starts without loading its command", u);
      if (WAITC && !rst && we && can_start[u] && start_slot[u] == wa)
        $fatal(1, "otpu_seq: unit %0d starts slot %0d before its command is written", u, wa);
    end
`endif
  end

`ifndef SYNTHESIS
  // shadow of c_fp / sfp in otpu_pkg form: the scoreboard is checked against conflict(), on
  // the footprints with MM's RMAX range folded into its output range as fp_seg does (ref), and
  // for coverage on the footprints as they are (raw)
  fp_t c_ref, c_raw;
  fp_t sfp_ref [WIN], sfp_raw [WIN];
  logic [7:0]  sidx [WIN];                   // WAITC: the slot's index among its unit's
  logic [31:0] spc [WIN];

  // fp_seg's hull in otpu_pkg form: two TMEM write ranges (only MM with RMAX) become one
  function automatic fp_t fp_hull(input fp_t f);
    fp_t o;
    o = f;
    if (f.wr[0].v && f.wr[0].sp == SP_TMEM && f.wr[1].v && f.wr[1].sp == SP_TMEM) begin
      o.wr[0].lo = (f.wr[1].lo < f.wr[0].lo) ? f.wr[1].lo : f.wr[0].lo;
      o.wr[0].hi = (f.wr[1].hi > f.wr[0].hi) ? f.wr[1].hi : f.wr[0].hi;
      o.wr[1].v = 1'b0;
    end
    return o;
  endfunction
`endif

  // the window's footprints: written at dispatch, read only for valid slots, so outside the
  // reset (250 MHz, FK35: otpu_slice to_err -> srst -> sfp's enable, 2 levels, 93% routing,
  // -0.55 ns)
  always_ff @(posedge clk) if (c_go) sfp[free_slot] <= c_fp;

  // readiness (for the profiler: when did the dependencies clear; ev.s_rdy). Outside the reset:
  // a slot's flag clears at its dispatch, so these 16 x 33 flip-flops' enables do not include the
  // slice's reset (250 MHz, FK35: otpu_slice to_err -> srst -> srdy_c's enable, 2 levels, 92%
  // routing, -0.70 ns)
  always_ff @(posedge clk) begin
    for (int i = 0; i < WIN; i++)
      if (sv[i] && !sready[i] && (WAITC ? w_all_q[i] : sdep[i] == '0)) begin
        sready[i] <= 1'b1;
        srdy_c[i] <= cyc;
      end
    if (c_go) sready[free_slot] <= 1'b0;
  end

  always_ff @(posedge clk) begin
    ustart <= '0;
    urel <= 1'b0;
    if (rst) begin
      uq_r <= '0;
      pc <= '0; sp <= '0; cyc <= '0;
      halted <= 1'b0; error <= 1'b0; stopping <= 1'b0;
      rl_wait <= 1'b0; ch_req <= 1'b0;
      p_v <= 1'b0; o_v <= 1'b0; s_v <= 1'b0; q_v <= 1'b0; f_v <= 1'b0; y_v <= 1'b0;
      k_v <= 1'b0; c_v <= 1'b0; cnt <= '0; lp <= 2'd0; at_end_q <= 1'b0;
      icount <= '0;
      sv <= '0; sstarted <= '0;
      for (int u = 0; u < NUNITS; u++) begin uq_h[u] <= '0; uq_t[u] <= '0; end
      for (int u = 0; u < NUNITS; u++) begin wd_n[u] <= '0; wc_n[u] <= '0; end
      vout <= '0;
      vnx_q <= 1'b0;
      for (int i = 0; i < WIN; i++) sdep[i] <= '0;
`ifndef SYNTHESIS
      sdone <= '0;
`endif
    end else begin
      cyc <= cyc + 1;
      if (ld) pc <= fa;
      // at_end for the instruction at fa, under the stack as this move leaves it: a LOOP that
      // enters pushes [pc + 1, pc + len) and fa is pc + 1; advance() at a loop end that pops
      // leaves the entry below on top; otherwise the top stays (decremented or not)
      if (ld)
        at_end_q <= (ir_v && op == OP_LOOP) ? (lp_nz ? lp_len1 : hit1) :
                    (ir_v && at_end && !t_gt1) ? hit2 : hit1;
`ifndef SYNTHESIS
      // the stack's top two entries in flip-flops are the arrays'
      if (ir_v && at_end != ((sp != 0) && stk_end[sp-1] == pc))
        $fatal(1, "otpu_seq: at_end %b at pc %0d, the stack's %b", at_end, pc,
               (sp != 0) && stk_end[sp-1] == pc);
      if (sp >= 1 && (t_start != stk_start[sp-1] || t_end != stk_end[sp-1] ||
                      t_rem != stk_rem[sp-1] || t_gt1 != (stk_rem[sp-1] > 1) ||
                      t_gt2 != (stk_rem[sp-1] > 2)))
        $fatal(1, "otpu_seq: the loop stack's top in flip-flops differs (sp %0d)", sp);
      if (sp >= 2 && (n_start != stk_start[sp-2] || n_end != stk_end[sp-2] ||
                      n_rem != stk_rem[sp-2] || n_gt1 != (stk_rem[sp-2] > 1)))
        $fatal(1, "otpu_seq: the loop stack's second entry in flip-flops differs (sp %0d)", sp);
`endif
`ifndef SYNTHESIS
      // fa is the pc the one-instruction-per-cycle fetch would move to (the next_pc rule)
      if (ld && ir_v &&
          fa != ((op == OP_LOOP) ? ((lp_cnt == 0) ? pc + 1 + lp_len : pc + 1) :
                 (at_end && stk_rem[sp-1] > 1) ? stk_start[sp-1] : pc + 1))
        $fatal(1, "otpu_seq: fetch address %0d does not follow pc %0d", fa, pc);
      // the segregated footprints give exactly otpu_pkg's conflict() and conflict_dram(); a
      // footprint whose TMEM ranges overflowed rt_t (all raised by fp_seg) conflicts with all.
      // The hull only adds dependencies: every conflict() of the raw footprints is kept
      if (c_v && !WAITC)
        for (int i = 0; i < WIN; i++) begin
          logic [1:0] cs;
          cs = conf_s(c_fp, sfp[i]);
          // K's registered dependencies are exactly those of C's footprint on the window now
          if (!hz_rob && !hz_wc && (ndep[i] != (sv[i] && !fin[i] && cs[1]) ||
                                    ndepd[i] != (sv[i] && !fin[i] && cs[0])))
            $fatal(1, "otpu_seq: pc %0d: registered dependency on slot %0d %b, conf_s %b", c_pc, i,
                   {ndep[i], ndepd[i]}, {sv[i] && !fin[i] && cs[1], sv[i] && !fin[i] && cs[0]});
          if (sv[i] && !hz_cs && (c_fp.all == c_ref.all && sfp[i].all == sfp_ref[i].all ?
                        cs != {conflict(c_ref, sfp_ref[i]), conflict_dram(c_ref, sfp_ref[i])} :
                        cs != 2'b11))
            $fatal(1, "otpu_seq: scoreboard differs from conflict() for pc %0d vs slot %0d",
                   c_pc, i);
          if (sv[i] && conflict(c_raw, sfp_raw[i]) && !cs[1])
            $fatal(1, "otpu_seq: scoreboard misses a conflict() of pc %0d vs slot %0d",
                   c_pc, i);
        end
      // fp_seg's assumptions: ACT ranges fit [alo, ahi), DRAM ranges share their write role
      if (q_v) begin
        logic dr, dwr;
        dr = 1'b0; dwr = 1'b0;
        for (int i = 0; i < 4; i++) begin
          if (q_fp.rd[i].v && q_fp.rd[i].sp == SP_DRAM) dr = 1'b1;
          if (q_fp.rd[i].v && q_fp.rd[i].sp == SP_ACT &&
              (q_fp.rd[i].lo[31:8] != 0 || q_fp.rd[i].hi[31:17] != 0))
            $fatal(1, "otpu_seq: ACT read range of pc %0d exceeds 8/17 bits", q_pc);
        end
        for (int i = 0; i < 3; i++) begin
          if (q_fp.wr[i].v && q_fp.wr[i].sp == SP_DRAM) dwr = 1'b1;
          if (q_fp.wr[i].v && q_fp.wr[i].sp == SP_ACT &&
              (q_fp.wr[i].lo[31:8] != 0 || q_fp.wr[i].hi[31:17] != 0))
            $fatal(1, "otpu_seq: ACT write range of pc %0d exceeds 8/17 bits", q_pc);
        end
        if (dr && dwr)
          $fatal(1, "otpu_seq: pc %0d both reads and writes DRAM (one dw bit)", q_pc);
      end
`endif
`ifndef SYNTHESIS
      // WAITC: the wait entry covers every dependency conflict() finds on the window (a slot
      // of the same unit, but the VPU, needs none: it started first); a VPU slot is at most VMW
      // back
      if (WAITC && c_go) begin
        for (int i = 0; i < WIN; i++)
          if (sv[i] && !fin[i]) begin
            int ui;
            ui = 0;
            for (int u = 0; u < NUNITS; u++) if (soh[i][u]) ui = u;
            if (ui == U_VPU && wc_vg > 0 && wc_vg <= VMW) begin
              logic [CW-1:0] rk;
              rk = wd_n[U_VPU] - sidx[i];
              if (rk == 0 || rk > CW'(wc_vg))
                $fatal(1, "otpu_seq: WAITC: pc %0d: VPU slot %0d (pc %0d) %0d back", c_pc, i,
                       spc[i], rk);
              if (conflict(c_ref, sfp_ref[i]) && !c_sync[28 + int'(rk) - 1])
                $fatal(1, "otpu_seq: WAITC: pc %0d's VPU mask %h misses slot %0d (pc %0d, %0d back)",
                       c_pc, c_sync[28 +: VMW], i, spc[i], rk);
            end
            for (int j = 0; j < NWT; j++)
              if (wunit(j) == ui && ui != int'(c_unit) && ui != U_VPU &&
                  ((j < NUNITS) ? conflict(c_ref, sfp_ref[i]) :
                   c_cmd.op == OP_MM && conflict_dram(c_ref, sfp_ref[i]))) begin
                logic [3:0] k;
                logic [CW-1:0] df;
                k = c_sync[4 * j +: 4];
                df = wd_n[ui] - CW'(k) - sidx[i];      // the producer's index - the slot's
                if (k == 0 || df[CW-1])
                  $fatal(1, "otpu_seq: WAITC: pc %0d's wait %0d (k %0d) misses slot %0d (pc %0d)",
                         c_pc, j, k, i, spc[i]);
              end
          end
        sidx[free_slot] <= wd_n[c_unit];
        spc[free_slot] <= c_pc;
      end
`endif
      icount <= icount + 32'(r_ret) + 32'(c_go);
      // completions
      if (can_rel) begin
        urel <= 1'b1;
        uq_r <= uq_r + 1;
      end
      for (int i = 0; i < WIN; i++) begin
        sdep[i] <= sdep[i] & ~fin;
        sdepd[i] <= sdepd[i] & ~fin;
        if (fin[i]) begin
          sv[i] <= 1'b0;
        end
      end
`ifndef SYNTHESIS
      // +hz_rob: a completed slot is freed once every older one is
      if (hz_rob)
        for (int i = 0; i < WIN; i++) begin
          logic dn;
          dn = sdone[i] || fin[i];
          sdone[i] <= dn;
          sv[i] <= sv[i] && !(dn && (older[i] & sv & ~(sdone | fin)) == '0);
        end
`endif
      for (int u = 0; u < NUNITS; u++) if (udone[u]) uq_h[u] <= uq_h[u] + 1;
`ifndef SYNTHESIS
      for (int i = 0; i < WIN; i++) vxd[i] <= vxd[i] & ~fin;
`endif
      if (WAITC) begin
        for (int u = 0; u < NUNITS; u++) if (udone[u] && u != U_VPU) wc_n[u] <= wc_n[u] + 1'b1;
        vout <= (vout & ~v_fin) |
                ((c_go && c_unit == U_VPU) ? VN'(1) << wd_n[U_VPU][VIW-1:0] : '0);
        // (a VPU dispatch sets bit wd_n and moves wd_n on: the next index's bit is wd_n + 2's)
        begin
          logic [VIW-1:0] i1, i2;
          i1 = VIW'(wd_n[U_VPU] + 1'b1);
          i2 = VIW'(wd_n[U_VPU] + 2'd2);
          vnx_q <= (c_go && c_unit == U_VPU) ? vout[i2] && !v_fin[i2] : vout[i1] && !v_fin[i1];
        end
        for (int i = 0; i < WIN; i++) vpend[i] <= vpend[i] & ~v_fin;
      end
      // starts
      for (int u = 0; u < NUNITS; u++) begin
`ifndef SYNTHESIS
        // the age matrix orders every pair of candidates: at most one oldest (none for a cycle
        // when the oldest of a cycle ago has just started)
        if (!$onehot0(sel[u]) || (|sel[u] && !cand_any[u]))
          $fatal(1, "otpu_seq: unit %0d oldest-candidate select %b is not one-hot", u, sel[u]);
`endif
        for (int i = 0; i < WIN; i++)
          if (can_start[u] && sel[u][i]) sstarted[i] <= 1'b1;
        if (can_start[u]) begin
          ustart[u] <= 1'b1;
          ucs[u] <= start_slot[u];
          uq[u][uq_t[u][SW-1:0]] <= start_slot[u];
          uq_t[u] <= uq_t[u] + 1;
        end
      end
      // ---- C: dispatch into the window
      if (c_go) begin
        sv[free_slot] <= 1'b1;
        older[free_slot] <= sv;           // every slot in the window is older
        for (int i = 0; i < WIN; i++) older[i][free_slot] <= 1'b0;
        soh[free_slot] <= NUNITS'(1) << c_unit;
        sstarted[free_slot] <= 1'b0;
        sdep[free_slot] <= ndep;
        sdepd[free_slot] <= ndepd;
        if (WAITC) begin
          logic [VN-1:0] m;
          for (int j = 0; j < NWT; j++) begin
            logic [3:0] k;
            k = c_sync[4 * j +: 4];
            wv[free_slot][j] <= k != 0 && wunit(j) != int'(c_unit) && wunit(j) != U_VPU;
            wt[free_slot][j] <= wd_n[wunit(j)] - CW'(k) + 1'b1;
          end
          // the VPU mask: bit r - 1 names the VPU instruction r back (index wd_n - r)
          m = '0;
          for (int r = 1; r <= VMW; r++)
            if (c_sync[28 + r - 1]) m[VIW'(wd_n[U_VPU] - CW'(r))] = 1'b1;
          vpend[free_slot] <= m & vout & ~v_fin;
          vix[free_slot] <= VIW'(wd_n[c_unit]);
          wd_n[c_unit] <= wd_n[c_unit] + 1'b1;
        end
`ifndef SYNTHESIS
        sdone[free_slot] <= 1'b0;
        sfpx[free_slot] <= c_fpx;
        vxi[free_slot] <= wd_n[c_unit];
        for (int j = 0; j < WIN; j++) begin
          logic [1:0] cf;
          cf = conf_s(c_fpx, sfpx[j]);
          vxd[free_slot][j] <= sv[j] && !fin[j] && soh[j][U_VPU] && cf[1];
        end
`endif
`ifndef SYNTHESIS
        sfp_ref[free_slot] <= c_ref;
        sfp_raw[free_slot] <= c_raw;
`endif
      end
      // ---- K -> C: C takes K's instruction and its dependencies (K's conf_s on the window and
      // on C's instruction, which takes free_slot now if it dispatches)
      if (!WAITC && kc) begin
        c_v <= k_v;
        if (k_v) begin
          c_cmd <= k_cmd; c_unit <= k_unit; c_pc <= k_pc; c_fp <= k_fp;
          for (int i = 0; i < WIN; i++) c_cf[i] <= conf_s(k_fp, sfp[i]);
          c_cx <= conf_s(k_fp, c_fp);
          c_xs <= free_slot;
          c_xg <= c_go;
`ifndef SYNTHESIS
          c_fpx <= fp_seg(k_cmd.op, footprint(k_cmd, D, S));
          c_ref <= fp_hull(footprint(k_cmd, D, S));
          c_raw <= footprint(k_cmd, D, S);
`endif
        end
      end
      // ---- the FIFO's head -> K
      if (k_pop) k_v <= 1'b1;
      else if (!WAITC && kc) k_v <= 1'b0;
      // ---- P -> O -> S -> Q -> F -> Y -> the FIFO: valid bits (the data stages below have no
      // enable); the credits: one in at P -> O, one out at the FIFO -> K
      if (!WAITC) begin
        o_v <= p_v; s_v <= o_v; q_v <= s_v; f_v <= q_v; y_v <= f_v;
        cnt <= cnt + 4'(p_v) - 4'(k_pop);
      end
      // ---- P (WAITC: the instruction goes to C)
      if (WAITC && p_adv) begin
        c_v <= p_v;
        if (p_v) begin
          c_cmd <= p_cmd; c_unit <= p_unit; c_pc <= p_pc; c_sync <= p_sync;
`ifndef SYNTHESIS
          c_ref <= fp_hull(footprint(p_cmd, D, S));
          c_raw <= footprint(p_cmd, D, S);
          c_fpx <= fp_seg(p_cmd.op, footprint(p_cmd, D, S));
`endif
        end
      end
      if (r_take || !WAITC) p_v <= 1'b0;      // (P -> O every cycle: R refills it below)
      // ---- RLD's value (R holds meanwhile, so nothing below writes R this cycle; the write is
      // in R's own process)
      if (rl_v) rl_wait <= 1'b0;
      // ---- R: fetch / execute control / hand over
      if (stopping) begin
        if (sv == '0 && pipe_empty) halted <= 1'b1;
      end else if (!halted && ir_v && !rl_wait) begin
        case (op)
          OP_NOP: advance();
          OP_HALT: begin
            stopping <= 1'b1;
            ch_req <= flags[HF_CHAIN];
            ch_addr <= rv(ra);
            ch_n <= rv(rb);
          end
          OP_LI, OP_ADDI: advance();   // (R[rd] in R's own process)
          OP_LOOP: begin
            if (lp == 2'd0) begin
              lp <= 2'd1;
              lp_cnt <= rv(ra) + iw[2];
              lp_len <= iw[1];
              lp_skip <= pc + 1 + iw[1];
            end else if (lp == 2'd1) begin
              lp <= 2'd2;
              lp_nz <= lp_cnt != '0;
              lp_gt1 <= lp_cnt > 32'd1;
              lp_gt2 <= lp_cnt > 32'd2;
              lp_len1 <= lp_len == 32'd1;
            end else begin
              lp <= 2'd0;
              if (lp_nz) begin
                stk_start[sp] <= pc + 1;
                stk_end[sp]   <= pc + lp_len;
                stk_rem[sp]   <= lp_cnt;
                sp <= sp + 1;
                n_start <= t_start; n_end <= t_end; n_rem <= t_rem; n_gt1 <= t_gt1;
                t_start <= pc + 1; t_end <= pc + lp_len; t_rem <= lp_cnt;
                t_gt1 <= lp_gt1; t_gt2 <= lp_gt2;
              end
            end
          end
          default: begin
            if (dunit < 0) begin
              error <= 1'b1;
              halted <= 1'b1;
            end else if (r_take) begin
              p_v <= 1'b1;
              p_cmd <= dcmd; p_unit <= 3'(dunit); p_pc <= pc; p_sync <= ir_sync;
              advance();
              if (op == OP_RLD) begin
                rl_wait <= 1'b1;
                rl_rd <= rd;
              end
            end
          end
        endcase
      end
    end
  end

`ifndef SYNTHESIS
  // ---- EXPERIMENTAL (docs/seq_vpu.md): wait counts, their oracle, and hazard statistics
  localparam int NPC = IMEM_WORDS / 8;     // instructions IMEM holds
  localparam int NCLS = 12;                // instruction classes of the statistics (hz_cls)
  localparam int KMAX = 16;                // ranks the oracle looks back, per unit
  // per pc: k for units 0..4 (any conflict), then the DRAM-only k for the DMA and the quantizer
  // (an MM may stream once those are clear); 0: no conflict within KMAX
  int  wc_k [NPC][7];
  int  hz_fd = 0, hz_kfd = 0;
  bit  hz_seen [NPC];
  int  hz_cnt [NPC];
  int  kmin [NPC][7];
  fps_t hist [NUNITS][KMAX];
  int  hist_n [NUNITS];
  int  scls [WIN];
  // statistics
  longint st_cyc, st_wfull, st_rlw, st_disp, st_nodep, st_vhold;
  longint st_u [NUNITS][6];                // busy, empty, start, unit not ready, blocked, (spare)
  longint st_by [NUNITS][NUNITS];          // blocked cycles of unit u by producer unit p
  longint st_bcls [NCLS][NCLS];            // blocked cycles by (waiting class, producer class)
  longint st_rel [NUNITS + 1];             // MXU started, release waits, by producer unit; total
  longint st_dep [NCLS][NCLS];             // dependencies at dispatch by (class, producer class)
  longint st_kind [6];                     // TMEM RAW, WAR, WAW, DRAM, ACT, all (BAR / WAITW)
  longint st_kh [NUNITS][NUNITS][KMAX + 1];// oracle ranks: consumer unit, producer unit, k
  longint st_nu [NUNITS + 1];              // instructions by the number of units they depend on
  longint st_wcmiss;

  function automatic int hz_cls(input logic [7:0] op, input logic [7:0] f);
    case (op)
      OP_LD: return 0;
      OP_ST: return 1;
      OP_DSTEP, OP_STREAM: return 2;
      OP_WAITW: return 3;
      OP_MM: return 4;
      OP_QACT: return 5;
      OP_QST: return 6;
      OP_VOP: return is_reduce(f) ? 8 : (f_cc(f) != CC_NONE) ? 9 : 7;
      OP_RLD: return 10;
      default: return 11;
    endcase
  endfunction

  // which kinds a dependency is: TMEM RAW, WAR, WAW, DRAM, ACT, all
  function automatic logic [5:0] hz_kind(input fps_t n, input fps_t e);
    logic [5:0] k;
    rt_t nr [5];
    rt_t er [5];
    k = '0;
    if (n.all || e.all) k[5] = 1'b1;
    nr[0] = n.t0.w ? '0 : n.t0.r; nr[1] = n.t1; nr[2] = n.t2; nr[3] = n.t3; nr[4] = n.t4;
    er[0] = e.t0.w ? '0 : e.t0.r; er[1] = e.t1; er[2] = e.t2; er[3] = e.t3; er[4] = e.t4;
    for (int j = 0; j < 5; j++) begin
      if (e.t0.w && ovr_t(nr[j], e.t0.r)) k[0] = 1'b1;
      if (n.t0.w && ovr_t(n.t0.r, er[j])) k[1] = 1'b1;
    end
    if (n.t0.w && e.t0.w && ovr_t(n.t0.r, e.t0.r)) k[2] = 1'b1;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        if ((n.dw || e.dw) && ovr(n.d[i], e.d[j])) k[3] = 1'b1;
    if ((n.aw || e.aw) && {9'd0, n.alo} < e.ahi && {9'd0, e.alo} < n.ahi) k[4] = 1'b1;
    return k;
  endfunction

  function automatic int hz_unit(input logic [NUNITS-1:0] oh);
    for (int u = 0; u < NUNITS; u++) if (oh[u]) return u;
    return 0;
  endfunction

  initial begin
    string fn;
    int fd, pc, r, n;
    int k [7];
    for (int i = 0; i < NPC; i++) begin
      hz_seen[i] = 1'b0; hz_cnt[i] = 0;
      for (int j = 0; j < 7; j++) begin wc_k[i][j] = 1; kmin[i][j] = 0; end
    end
    for (int u = 0; u < NUNITS; u++) hist_n[u] = 0;
    if ($value$plusargs("hz_wc=%s", fn)) begin
      fd = $fopen(fn, "r");
      if (fd == 0) $fatal(1, "otpu_seq: +hz_wc=%s: cannot open", fn);
      while (!$feof(fd)) begin
        r = $fscanf(fd, "%d %d %d %d %d %d %d %d %d\n", pc, k[0], k[1], k[2], k[3], k[4], k[5],
                    k[6], n);
        if (r >= 8 && pc >= 0 && pc < NPC)
          for (int j = 0; j < 7; j++)
            wc_k[pc][j] = (hz_kclamp > 0 && k[j] > hz_kclamp) ? hz_kclamp : k[j];
      end
      $fclose(fd);
      hz_wc = 1'b1;
      if (!hz_ino) $fatal(1, "otpu_seq: +hz_wc needs +hz_ino=1 (in-order units)");
    end
    if ($value$plusargs("hz_kout=%s", fn)) hz_kfd = $fopen(fn, "w");
    if ($value$plusargs("hz_log=%s", fn)) hz_fd = $fopen(fn, "w");
    st_cyc = 0; st_wfull = 0; st_rlw = 0; st_disp = 0; st_nodep = 0; st_wcmiss = 0; st_vhold = 0;
    for (int u = 0; u < NUNITS; u++) begin
      for (int j = 0; j < 6; j++) st_u[u][j] = 0;
      for (int p = 0; p < NUNITS; p++) begin
        st_by[u][p] = 0;
        for (int j = 0; j <= KMAX; j++) st_kh[u][p][j] = 0;
      end
    end
    for (int u = 0; u <= NUNITS; u++) begin st_rel[u] = 0; st_nu[u] = 0; end
    for (int a = 0; a < NCLS; a++)
      for (int b = 0; b < NCLS; b++) begin st_bcls[a][b] = 0; st_dep[a][b] = 0; end
    for (int j = 0; j < 6; j++) st_kind[j] = 0;
  end

  // +hz_wc: slot i is a dependency when it is one of the older unit-u instructions beyond the
  // k_u - 1 most recent (in-order units complete in program order, so these are outstanding)
  always_comb begin
    wc_dep = '0; wc_depd = '0;
    if (hz_wc && c_v)
      for (int i = 0; i < WIN; i++)
        if (sv[i] && !fin[i]) begin
          int u, y, k, kd;
          u = hz_unit(soh[i]);
          y = 0;
          for (int j = 0; j < WIN; j++) y += int'(sv[j] && soh[j][u] && older[j][i]);
          k = wc_k[c_pc % NPC][u];
          kd = (u == U_DMA) ? wc_k[c_pc % NPC][5] : (u == U_Q) ? wc_k[c_pc % NPC][6] : 0;
          if (u == int'(c_unit) && u != U_VPU) begin k = 0; kd = 0; end   // started first
          wc_dep[i] = k != 0 && y >= k - 1;
          wc_depd[i] = (c_cmd.op == OP_MM) ? (kd != 0 && y >= kd - 1) : wc_dep[i];
          if (u == U_VPU && (hz_wcvx == 1 || (hz_wcvx == 2 && c_unit == U_VPU) ||
                             (hz_wcvx == 3 && c_unit != U_VPU))) begin
            logic [1:0] cf;
            cf = conf_s(c_fp, sfp[i]);
            wc_dep[i] = cf[1];
            wc_depd[i] = cf[0];
          end
        end
  end

  always @(posedge clk) begin
    if (rst) begin
      for (int u = 0; u < NUNITS; u++) hist_n[u] <= 0;
    end else begin
      // ---- at dispatch: the oracle's ranks, the dependencies' classes and kinds
      if (c_go && (hz_fd != 0 || hz_kfd != 0)) begin
        int cu, cc, nu, pc;
        int kk [NUNITS];
        int kd [NUNITS];
        logic [NUNITS-1:0] du;
        cu = int'(c_unit);
        cc = hz_cls(c_cmd.op, c_cmd.w6[23:16]);
        scls[free_slot] <= cc;
        st_disp++;
        du = '0;
        for (int i = 0; i < WIN; i++)
          if (ndep[i]) begin
            logic [5:0] kn;
            kn = hz_kind(c_fpx, sfp[i]);
            st_dep[cc][scls[i]]++;
            for (int j = 0; j < 6; j++) if (kn[j]) st_kind[j]++;
            du |= soh[i];
          end
        if (ndep == '0) st_nodep++;
        nu = $countones(du);
        st_nu[nu]++;
        if (hz_wc)
          for (int i = 0; i < WIN; i++) begin
            logic [1:0] cf;
            cf = conf_s(c_fpx, sfp[i]);
            if (sv[i] && !fin[i] && cf[1] && !ndep[i] && !(soh[i][cu] && cu != U_VPU))
              st_wcmiss++;
          end
        for (int u = 0; u < NUNITS; u++) begin
          kk[u] = 0; kd[u] = 0;
          for (int j = 0; j < KMAX; j++)
            if (j < hist_n[u]) begin
              logic [1:0] cf;
              cf = conf_s(hz_kc ? c_fp : c_fpx, hist[u][j]);
              if (cf[1] && kk[u] == 0) kk[u] = j + 1;
              if (cf[0] && kd[u] == 0) kd[u] = j + 1;
            end
          st_kh[cu][u][kk[u]]++;
        end
        pc = int'(c_pc % NPC);
        for (int u = 0; u < NUNITS; u++)
          if (kk[u] != 0 && (kmin[pc][u] == 0 || kk[u] < kmin[pc][u])) kmin[pc][u] = kk[u];
        if (kd[U_DMA] != 0 && (kmin[pc][5] == 0 || kd[U_DMA] < kmin[pc][5]))
          kmin[pc][5] = kd[U_DMA];
        if (kd[U_Q] != 0 && (kmin[pc][6] == 0 || kd[U_Q] < kmin[pc][6])) kmin[pc][6] = kd[U_Q];
        hz_seen[pc] = 1'b1;
        hz_cnt[pc]++;
        for (int j = KMAX - 1; j > 0; j--) hist[cu][j] <= hist[cu][j - 1];
        hist[cu][0] <= hz_kc ? c_fp : c_fpx;
        hist_n[cu] <= (hist_n[cu] < KMAX) ? hist_n[cu] + 1 : KMAX;
      end
      // ---- every cycle of the run: what each unit does, and what a blocked one waits for
      if (!halted && hz_fd != 0) begin
        st_cyc++;
        if (c_v && !have_free) st_wfull++;
        if (c_v && have_free && wc_hold) st_vhold++;   // WAITC: a VOP held for the one VMW back
        if (rl_wait) st_rlw++;
        for (int u = 0; u < NUNITS; u++) begin
          logic [WIN-1:0] pend;
          for (int i = 0; i < WIN; i++) pend[i] = sv[i] && !sstarted[i] && soh[i][u];
          if (uq_h[u] != uq_t[u]) st_u[u][0]++;
          else if (pend == '0) st_u[u][1]++;
          else if (can_start[u]) st_u[u][2]++;
          else if (!urdy[u]) st_u[u][3]++;
          else begin
            int o;
            logic [WIN-1:0] b;
            st_u[u][4]++;
            o = 0;
            for (int i = 0; i < WIN; i++) if (pend[i] && (older[i] & pend) == '0) o = i;
            b = ((u == U_MXU) ? sdepd[o] : sdep[o]) & ~same_started[u];
            for (int p = 0; p < NUNITS; p++) begin
              logic hit;
              hit = 1'b0;
              for (int i = 0; i < WIN; i++) if (b[i] && soh[i][p]) hit = 1'b1;
              if (hit) st_by[u][p]++;
            end
            for (int i = 0; i < WIN; i++) if (b[i]) st_bcls[scls[o]][scls[i]]++;
          end
        end
        if (uq_r != uq_t[U_MXU] && !can_rel) begin
          logic [WIN-1:0] b;
          b = sdep[rel_slot] & ~same_started[U_MXU];
          st_rel[NUNITS]++;
          for (int p = 0; p < NUNITS; p++) begin
            logic hit;
            hit = 1'b0;
            for (int i = 0; i < WIN; i++) if (b[i] && soh[i][p]) hit = 1'b1;
            if (hit) st_rel[p]++;
          end
        end
      end
    end
  end

  final begin
    if (hz_kfd != 0) begin
      for (int pc = 0; pc < NPC; pc++)
        if (hz_seen[pc])
          $fdisplay(hz_kfd, "%0d %0d %0d %0d %0d %0d %0d %0d %0d", pc, kmin[pc][0], kmin[pc][1],
                    kmin[pc][2], kmin[pc][3], kmin[pc][4], kmin[pc][5], kmin[pc][6], hz_cnt[pc]);
      $fclose(hz_kfd);
    end
    if (hz_fd != 0) begin
      $fdisplay(hz_fd, "HZ cycles %0d dispatched %0d nodep %0d wfull %0d rlwait %0d wcmiss %0d vhold %0d",
                st_cyc, st_disp, st_nodep, st_wfull, st_rlw, st_wcmiss, st_vhold);
      $fdisplay(hz_fd, "HZ knobs tg %0d dg %0d rh %0d dh %0d win %0d ino %0d rob %0d wc %0d uq %0d,%0d,%0d,%0d,%0d",
                hz_tg, hz_dg, hz_rh, hz_dh, hz_win, hz_ino, hz_rob, hz_wc, hz_uq[0], hz_uq[1],
                hz_uq[2], hz_uq[3], hz_uq[4]);
      for (int u = 0; u < NUNITS; u++)
        $fdisplay(hz_fd, "HZU %0d busy %0d empty %0d start %0d notrdy %0d blocked %0d by %0d %0d %0d %0d %0d",
                  u, st_u[u][0], st_u[u][1], st_u[u][2], st_u[u][3], st_u[u][4], st_by[u][0],
                  st_by[u][1], st_by[u][2], st_by[u][3], st_by[u][4]);
      $fdisplay(hz_fd, "HZREL total %0d by %0d %0d %0d %0d %0d", st_rel[NUNITS], st_rel[0],
                st_rel[1], st_rel[2], st_rel[3], st_rel[4]);
      $fdisplay(hz_fd, "HZKIND tmem_raw %0d tmem_war %0d tmem_waw %0d dram %0d act %0d all %0d",
                st_kind[0], st_kind[1], st_kind[2], st_kind[3], st_kind[4], st_kind[5]);
      $fdisplay(hz_fd, "HZNU %0d %0d %0d %0d %0d %0d", st_nu[0], st_nu[1], st_nu[2], st_nu[3],
                st_nu[4], st_nu[5]);
      for (int a = 0; a < NCLS; a++)
        for (int b = 0; b < NCLS; b++) begin
          if (st_dep[a][b] != 0) $fdisplay(hz_fd, "HZDEP %0d %0d %0d", a, b, st_dep[a][b]);
          if (st_bcls[a][b] != 0) $fdisplay(hz_fd, "HZBLK %0d %0d %0d", a, b, st_bcls[a][b]);
        end
      for (int u = 0; u < NUNITS; u++)
        for (int p = 0; p < NUNITS; p++) begin
          string l;
          l = $sformatf("HZK %0d %0d", u, p);
          for (int j = 0; j <= KMAX; j++) l = {l, $sformatf(" %0d", st_kh[u][p][j])};
          $fdisplay(hz_fd, "%s", l);
        end
      $fclose(hz_fd);
    end
  end
`endif

  // ---- trace and activity events (otpu_pkg seq_ev_t): this cycle's, registered. A start's
  // ready cycle is looked up a cycle late, from the start's registered slot (ucs): if the slot
  // was not ready when it started, it either became ready at that cycle's edge (srdy_c = the
  // start cycle) or is still not ready (the start cycle, ev.cyc) -- the value at the start.
  seq_ev_t evq;
  always_ff @(posedge clk) begin
    evq.cyc <= cyc;
    evq.ret <= rst ? 2'd0 : 2'(r_ret) + 2'(c_go);
    evq.d <= !rst && c_go;
    evq.d_slot <= 5'(free_slot);
    evq.d_pc <= c_pc;
    evq.d_op <= c_cmd.op;
    evq.d_w1 <= c_cmd.w1; evq.d_w2 <= c_cmd.w2; evq.d_w3 <= c_cmd.w3;
    evq.g <= !rst && can_rel;
    evq.g_slot <= 5'(rel_slot);
    evq.e <= rst ? '0 : 32'(fin);
    for (int u = 0; u < NUNITS; u++) evq.busy[u] <= !rst && uq_h[u] != uq_t[u];
  end
  always_comb begin
    ev = evq;
    ev.s = ustart;
    for (int u = 0; u < NUNITS; u++) begin
      ev.s_slot[u] = 5'(ucs[u]);
      ev.s_rdy[u] = sready[ucs[u]] ? srdy_c[ucs[u]] : evq.cyc;
    end
  end

`ifndef SYNTHESIS
  initial if (WIN > 32) $fatal(1, "otpu_seq: the trace events hold 32 window slots");
  // WAITC: a count of 4 bits names a unit's 15 last instructions; an in-order unit's 16th last
  // has completed only if the window holds no more than 16
  initial if (WAITC && WIN > 16) $fatal(1, "otpu_seq: WAITC needs WIN <= 16 (%0d)", WIN);
  string dir;
  logic [31:0] init_w [IMEM_WORDS];
  initial begin
    for (int i = 0; i < IMEM_WORDS; i++) init_w[i] = '0;
    if ($value$plusargs("dir=%s", dir) && !$test$plusargs("boot"))
      $readmemh($sformatf("%s/prog_%0d.hex", dir, SID), init_w);
    for (int r = 0; r < NROW; r++)
      for (int w = 0; w < 8 * IPR; w++)
        u_imem.g_auto.mem[r][32 * w +: 32] = init_w[r * 8 * IPR + w];
  end
`endif
endmodule
