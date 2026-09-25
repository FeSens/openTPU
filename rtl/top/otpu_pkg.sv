// openTPU ISA constants, the decoded command passed from the sequencer to the units, and the
// memory footprints the sequencer's scoreboard uses to let units run concurrently.
package otpu_pkg;
  localparam logic [7:0] OP_NOP = 8'h00, OP_HALT = 8'h01, OP_LI = 8'h02, OP_ADDI = 8'h03,
                         OP_LOOP = 8'h04, OP_BAR = 8'h05, OP_LD = 8'h10, OP_ST = 8'h11,
                         OP_MM = 8'h20, OP_QACT = 8'h21, OP_QST = 8'h22, OP_VOP = 8'h30,
                         OP_GATHER = 8'h40;

  localparam logic [7:0] V_ADD = 0, V_SUB = 1, V_RSUB = 2, V_MUL = 3, V_MAX = 4, V_MIN = 5,
                         V_OUTER = 6, V_COPY = 8, V_EXP2 = 9, V_RECIP = 10, V_RSQRT = 11,
                         V_ABS = 12, V_FILL = 13, V_EXP2SUB = 14, V_LOG2 = 15, V_RSUM = 16,
                         V_RMAX = 17, V_RSSQ = 18, V_RDOT = 19;
  // VOP OUTER flags: one decay word T[d] for all columns / decay 1.0 (T[d] not read)
  localparam int VF_DSCALAR = 0, VF_DONE = 1;

  localparam logic [1:0] B_FULL = 0, B_ROW = 1, B_COL = 2, B_SCALAR = 3;

  // Execution units. Each runs its own instructions in program order; different units run
  // concurrently whenever their memory footprints do not conflict.
  localparam int U_DMA = 0, U_MXU = 1, U_Q = 2, U_VPU = 3, U_COLL = 4, NUNITS = 5;

  // Instruction with register-relative fields already resolved: w1 += R[ra], w2 += R[rb],
  // w3 += R[rc] (docs/isa.md).
  typedef struct packed {
    logic [7:0]  op;
    logic [7:0]  flags;
    logic [31:0] w1, w2, w3, w4, w5, w6, w7;
  } cmd_t;

  // Address ranges [lo, hi) in one of three spaces.
  localparam logic [1:0] SP_TMEM = 0, SP_DRAM = 1, SP_ACT = 2;
  typedef struct packed {
    logic        v;
    logic [1:0]  sp;
    logic [31:0] lo, hi;
  } rng_t;
  typedef struct packed {
    logic            all;       // BAR: conflicts with everything
    rng_t [3:0]      rd;
    rng_t [1:0]      wr;
  } fp_t;

  function automatic int unit_of(input logic [7:0] op);
    case (op)
      OP_LD, OP_ST:      return U_DMA;
      OP_MM:             return U_MXU;
      OP_QACT, OP_QST:   return U_Q;
      OP_VOP:            return U_VPU;
      OP_BAR, OP_GATHER: return U_COLL;
      default:           return -1;
    endcase
  endfunction

  function automatic rng_t mk(input logic [1:0] sp, input logic [31:0] lo, input logic [31:0] len);
    rng_t r;
    r.v = (len != 0);
    r.sp = sp;
    r.lo = lo;
    r.hi = lo + len;
    return r;
  endfunction

  function automatic logic ov(input rng_t a, input rng_t b);
    return a.v && b.v && a.sp == b.sp && a.lo < b.hi && b.lo < a.hi;
  endfunction

  // RAW, WAR or WAW between two instructions.
  function automatic logic conflict(input fp_t n, input fp_t e);
    if (n.all || e.all) return 1'b1;
    for (int i = 0; i < 2; i++) begin
      for (int j = 0; j < 4; j++)
        if (ov(n.wr[i], e.rd[j]) || ov(e.wr[i], n.rd[j])) return 1'b1;
      for (int j = 0; j < 2; j++)
        if (ov(n.wr[i], e.wr[j])) return 1'b1;
    end
    return 1'b0;
  endfunction

  // Like conflict(), restricted to DRAM ranges (an MM may stream once these are clear).
  function automatic logic conflict_dram(input fp_t n, input fp_t e);
    fp_t nd, ed;
    if (n.all || e.all) return 1'b1;
    nd = n; ed = e;
    for (int i = 0; i < 4; i++) begin
      if (nd.rd[i].sp != SP_DRAM) nd.rd[i].v = 1'b0;
      if (ed.rd[i].sp != SP_DRAM) ed.rd[i].v = 1'b0;
    end
    for (int i = 0; i < 2; i++) begin
      if (nd.wr[i].sp != SP_DRAM) nd.wr[i].v = 1'b0;
      if (ed.wr[i].sp != SP_DRAM) ed.wr[i].v = 1'b0;
    end
    return conflict(nd, ed);
  endfunction

  function automatic logic is_binary(input logic [7:0] f);
    return f == V_ADD || f == V_SUB || f == V_RSUB || f == V_MUL || f == V_MAX || f == V_MIN ||
           f == V_FILL || f == V_EXP2SUB;
  endfunction

  // the functions that read operand B in its bmode (OUTER's is always B_ROW)
  function automatic logic reads_b(input logic [7:0] f);
    return is_binary(f) || f == V_RDOT || f == V_OUTER;
  endfunction

  function automatic logic is_reduce(input logic [7:0] f);
    return f == V_RSUM || f == V_RMAX || f == V_RSSQ || f == V_RDOT;
  endfunction

  // Everything an instruction may read or write (docs/isa.md), conservatively as intervals.
  // Two steps so that hardware can register in between: fp_prod (the multiplications) and
  // fp_ranges (the intervals); footprint() is their composition.
  typedef struct packed { logic [31:0] p0, p1, p2, p3; } fpm_t;
  // the multiplications as (a, w) pairs; a < 2^24 whenever the product is used
  typedef struct packed { logic [3:0][23:0] a; logic [3:0][31:0] w; } fpo_t;
  // partial products: low 32 bits of a * w[15:0], low 16 bits of a * w[31:16] (one DSP each)
  typedef struct packed { logic [3:0][31:0] pl; logic [3:0][15:0] ph; } fpp_t;

  function automatic fpo_t fp_ops(input cmd_t c, input int D, input int S);
    fpo_t o;
    logic [31:0] rows, n, kb, m;
    o = '0;
    case (c.op)
      OP_MM: begin
        n = 32'(c.w4[15:0]); m = 32'(c.w6[23:16]);
        o.a[0] = 24'(n - 1); o.w[0] = c.w5;
        o.a[1] = 24'(n - 1); o.w[1] = c.w7;
        o.a[2] = 24'(m - 1); o.w[2] = 32'(c.w6[15:0]);
        o.a[3] = 24'(m);     o.w[3] = 32'(c.w6[15:0]);
      end
      OP_QACT: begin
        rows = 32'(c.w2[7:0]);
        o.a[0] = 24'(rows - 1); o.w[0] = c.w3;
      end
      OP_QST: begin
        rows = 32'(c.w4[15:0]); kb = 32'(c.w4[31:16]);
        o.a[0] = 24'(rows - 1);   o.w[0] = c.w5;
        o.a[1] = 24'(rows - 1);   o.w[1] = c.w6;
        o.a[2] = 24'(kb * D - 1); o.w[2] = c.w7;
        o.a[3] = 24'(rows);       o.w[3] = kb;
      end
      OP_VOP: begin
        rows = 32'(c.w4[15:0]);
        o.a[0] = 24'(rows - 1); o.w[0] = 32'(c.w5[31:16]);
        o.a[1] = 24'(rows - 1); o.w[1] = 32'(c.w6[15:0]);
        o.a[2] = 24'(rows - 1); o.w[2] = 32'(c.w5[15:0]);
      end
      OP_GATHER: begin
        rows = 32'(c.w3[15:0]);
        o.a[0] = 24'(rows - 1); o.w[0] = c.w4;
        o.a[1] = 24'(S - 1);    o.w[1] = c.w6;
        o.a[2] = 24'(rows - 1); o.w[2] = c.w5;
      end
      default: ;
    endcase
    return o;
  endfunction

  function automatic fpp_t fp_part(input fpo_t o);
    fpp_t p;
    for (int i = 0; i < 4; i++) begin
      p.pl[i] = 32'({8'd0, o.a[i]} * {16'd0, o.w[i][15:0]});
      p.ph[i] = 16'({8'd0, o.a[i]} * {16'd0, o.w[i][31:16]});
    end
    return p;
  endfunction

  function automatic fpm_t fp_sum(input fpp_t p);
    fpm_t m;
    m.p0 = p.pl[0] + {p.ph[0], 16'd0};
    m.p1 = p.pl[1] + {p.ph[1], 16'd0};
    m.p2 = p.pl[2] + {p.ph[2], 16'd0};
    m.p3 = p.pl[3] + {p.ph[3], 16'd0};
    return m;
  endfunction

  function automatic fpm_t fp_prod(input cmd_t c, input int D, input int S);
    return fp_sum(fp_part(fp_ops(c, D, S)));
  endfunction

  function automatic fp_t fp_ranges(input cmd_t c, input fpm_t p, input int D, input int S);
    fp_t f;
    logic [31:0] rows, cols, n, kb, m;
    f = '0;
    case (c.op)
      OP_LD: begin
        f.rd[0] = mk(SP_DRAM, c.w1, c.w3 << 2);
        f.wr[0] = mk(SP_TMEM, c.w2, c.w3);
      end
      OP_ST: begin
        f.rd[0] = mk(SP_TMEM, c.w2, c.w3);
        f.wr[0] = mk(SP_DRAM, c.w1, c.w3 << 2);
      end
      OP_MM: begin
        n = 32'(c.w4[15:0]); kb = 32'(c.w4[31:16]); m = 32'(c.w6[23:16]);
        if (n != 0 && kb != 0 && m != 0) begin
          f.rd[0] = mk(SP_DRAM, c.w1, p.p0 + kb * D);
          if (!c.flags[0]) f.rd[1] = mk(SP_DRAM, c.w2, p.p1 + kb * 4);
          if (c.flags[3])  f.rd[3] = mk(SP_TMEM, c.w2, m);                         // ASCALE
          f.rd[2] = mk(SP_ACT, 32'(c.w6[31:24]), kb);
          f.wr[0] = mk(SP_TMEM, c.w3, p.p2 + n);
          if (c.flags[2]) f.wr[1] = mk(SP_TMEM, c.w3 + p.p3, m);                   // RMAX
        end
      end
      OP_QACT: begin
        rows = 32'(c.w2[7:0]); kb = 32'(c.w2[31:16]);
        if (rows != 0 && kb != 0) begin
          f.rd[0] = mk(SP_TMEM, c.w1, p.p0 + kb * D);
          if (c.flags[1]) f.rd[1] = mk(SP_TMEM, c.w4, kb * D);                     // CSCALE
          if (c.flags[2]) f.rd[2] = mk(SP_TMEM, c.w5, rows);                       // RSCALE
          f.wr[0] = mk(SP_ACT, 32'(c.w2[15:8]), kb);
        end
      end
      OP_QST: begin
        rows = 32'(c.w4[15:0]); kb = 32'(c.w4[31:16]);
        if (rows != 0 && kb != 0) begin
          f.rd[0] = mk(SP_TMEM, c.w1, p.p0 + kb * D);
          f.wr[0] = mk(SP_DRAM, c.w2, p.p1 + p.p2 + 1);
          f.wr[1] = mk(SP_DRAM, c.w3, 4 * (c.flags[0] ? rows : p.p3));
        end
      end
      OP_VOP: begin
        rows = 32'(c.w4[15:0]); cols = 32'(c.w4[31:16]);
        if (rows != 0 && cols != 0) begin
          // OUTER reads and writes dst in place (the write range covers the read); its A field
          // is the decay (none with DONE) and w7 the column vector
          if (c.w6[23:16] == V_OUTER) begin
            if (!c.flags[VF_DONE]) f.rd[0] = mk(SP_TMEM, c.w2, c.flags[VF_DSCALAR] ? 1 : cols);
            f.rd[2] = mk(SP_TMEM, c.w7, cols);
          end else if (c.w6[23:16] != V_FILL)
            f.rd[0] = mk(SP_TMEM, c.w2, p.p0 + cols);
          if (reads_b(c.w6[23:16])) begin
            case (c.w6[25:24])
              B_FULL: f.rd[1] = mk(SP_TMEM, c.w3, p.p1 + cols);
              B_ROW:  f.rd[1] = mk(SP_TMEM, c.w3, p.p1 + 1);
              B_COL:  f.rd[1] = mk(SP_TMEM, c.w3, cols);
              default: ;
            endcase
          end
          if (is_reduce(c.w6[23:16]))
            f.wr[0] = mk(SP_TMEM, c.w1, p.p2 + 1);
          else
            f.wr[0] = mk(SP_TMEM, c.w1, p.p2 + cols);
        end
      end
      OP_GATHER: begin
        rows = 32'(c.w3[15:0]); cols = 32'(c.w3[31:16]);
        if (rows != 0 && cols != 0) begin
          f.rd[0] = mk(SP_TMEM, c.w1, p.p0 + cols);
          f.wr[0] = mk(SP_TMEM, c.w2, p.p1 + p.p2 + cols);
        end
      end
      OP_BAR: f.all = 1'b1;
      default: ;
    endcase
    return f;
  endfunction

  function automatic fp_t footprint(input cmd_t c, input int D, input int S);
    return fp_ranges(c, fp_prod(c, D, S), D, S);
  endfunction

  // ---- activity and trace events (docs/observability.md). Every field describes cycle `cyc`
  // and is reported one cycle later, from flip-flops or from sums of flip-flops. Slot fields
  // hold up to 32 window slots.
  // The sequencer's part: dispatches (D), starts (S), MM releases (G), completions (E).
  typedef struct packed {
    logic [31:0]             cyc;       // the sequencer's cycle counter (0 = the first cycle out of reset)
    logic                    d;         // dispatched: pc into slot d_slot, resolved op and w1..w3
    logic [4:0]              d_slot;
    logic [31:0]             d_pc;
    logic [7:0]              d_op;
    logic [31:0]             d_w1, d_w2, d_w3;
    logic [NUNITS-1:0]       s;         // unit u started slot s_slot[u], ready since s_rdy[u]
    logic [NUNITS-1:0][4:0]  s_slot;
    logic [NUNITS-1:0][31:0] s_rdy;
    logic                    g;         // the MXU was released on slot g_slot
    logic [4:0]              g_slot;
    logic [31:0]             e;         // slots completed
    logic [NUNITS-1:0]       busy;      // the unit has a started, uncompleted instruction
    logic [1:0]              ret;       // instructions retired
  } seq_ev_t;

  // Per-window (P, Q) and per-run (H) counter fields, in trace-line order
  localparam int NP = 9, NQ = 6, NH = 4;
  // The slice's events: the sequencer's, the units' counters at an instruction's end (U), the
  // P/Q window sums (w: a window of w_n cycles ended at cyc) and the port totals at the halt (h).
  typedef struct packed {
    seq_ev_t                 sq;
    logic                    mac;       // the MXU consumed a weight chunk
    logic                    deny;      // a unit with TMEM requests was not granted
    logic                    u_mxu;     // MXU instruction ended: starve, bp, frz, deny
    logic [3:0][31:0]        u_mxu_v;   // [0] starve ... [3] deny
    logic                    u_q;       // QACT ended: frz
    logic [31:0]             u_q_frz;
    logic                    u_vpu;     // VPU instruction ended: frz
    logic [31:0]             u_vpu_frz;
    logic                    w;
    logic [31:0]             w_n;
    logic [NP-1:0][31:0]     w_p;       // bm bd am aq mx fm fq fv fc
    logic [NQ-1:0][31:0]     w_q;       // bs as ms mb ff ld
    logic                    h;
    logic [NH-1:0][31:0]     h_v;       // bmxu bdma amxu aq
  } perf_t;
endpackage
