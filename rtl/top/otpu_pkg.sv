// openTPU ISA constants, the decoded command passed from the sequencer to the units, and the
// memory footprints the sequencer's scoreboard uses to let units run concurrently.
package otpu_pkg;
  localparam logic [7:0] OP_NOP = 8'h00, OP_HALT = 8'h01, OP_LI = 8'h02, OP_ADDI = 8'h03,
                         OP_LOOP = 8'h04, OP_BAR = 8'h05, OP_LD = 8'h10, OP_ST = 8'h11,
                         OP_MM = 8'h20, OP_QACT = 8'h21, OP_QST = 8'h22, OP_VOP = 8'h30,
                         OP_GATHER = 8'h40;

  localparam logic [7:0] V_ADD = 0, V_SUB = 1, V_RSUB = 2, V_MUL = 3, V_MAX = 4, V_MIN = 5,
                         V_COPY = 8, V_EXP2 = 9, V_RECIP = 10, V_RSQRT = 11, V_ABS = 12,
                         V_FILL = 13, V_EXP2SUB = 14, V_RSUM = 16, V_RMAX = 17,
                         V_RSSQ = 18;

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

  // Everything an instruction may read or write (docs/isa.md), conservatively as intervals.
  function automatic fp_t footprint(input cmd_t c, input int D, input int S);
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
          f.rd[0] = mk(SP_DRAM, c.w1, (n - 1) * c.w5 + kb * D);
          if (!c.flags[0]) f.rd[1] = mk(SP_DRAM, c.w2, (n - 1) * c.w7 + kb * 4);
          if (c.flags[3])  f.rd[3] = mk(SP_TMEM, c.w2, m);                         // ASCALE
          f.rd[2] = mk(SP_ACT, 32'(c.w6[31:24]), kb);
          f.wr[0] = mk(SP_TMEM, c.w3, (m - 1) * 32'(c.w6[15:0]) + n);
          if (c.flags[2]) f.wr[1] = mk(SP_TMEM, c.w3 + m * 32'(c.w6[15:0]), m);   // RMAX
        end
      end
      OP_QACT: begin
        rows = 32'(c.w2[7:0]); kb = 32'(c.w2[31:16]);
        if (rows != 0 && kb != 0) begin
          f.rd[0] = mk(SP_TMEM, c.w1, (rows - 1) * c.w3 + kb * D);
          if (c.flags[1]) f.rd[1] = mk(SP_TMEM, c.w4, kb * D);                     // CSCALE
          if (c.flags[2]) f.rd[2] = mk(SP_TMEM, c.w5, rows);                       // RSCALE
          f.wr[0] = mk(SP_ACT, 32'(c.w2[15:8]), kb);
        end
      end
      OP_QST: begin
        rows = 32'(c.w4[15:0]); kb = 32'(c.w4[31:16]);
        if (rows != 0 && kb != 0) begin
          f.rd[0] = mk(SP_TMEM, c.w1, (rows - 1) * c.w5 + kb * D);
          f.wr[0] = mk(SP_DRAM, c.w2, (rows - 1) * c.w6 + (kb * D - 1) * c.w7 + 1);
          f.wr[1] = mk(SP_DRAM, c.w3, 4 * (c.flags[0] ? rows : rows * kb));
        end
      end
      OP_VOP: begin
        rows = 32'(c.w4[15:0]); cols = 32'(c.w4[31:16]);
        if (rows != 0 && cols != 0) begin
          if (c.w6[23:16] != V_FILL)
            f.rd[0] = mk(SP_TMEM, c.w2, (rows - 1) * 32'(c.w5[31:16]) + cols);
          if (is_binary(c.w6[23:16])) begin
            case (c.w6[25:24])
              B_FULL: f.rd[1] = mk(SP_TMEM, c.w3, (rows - 1) * 32'(c.w6[15:0]) + cols);
              B_ROW:  f.rd[1] = mk(SP_TMEM, c.w3, (rows - 1) * 32'(c.w6[15:0]) + 1);
              B_COL:  f.rd[1] = mk(SP_TMEM, c.w3, cols);
              default: ;
            endcase
          end
          if (c.w6[23:16] == V_RSUM || c.w6[23:16] == V_RMAX || c.w6[23:16] == V_RSSQ)
            f.wr[0] = mk(SP_TMEM, c.w1, (rows - 1) * 32'(c.w5[15:0]) + 1);
          else
            f.wr[0] = mk(SP_TMEM, c.w1, (rows - 1) * 32'(c.w5[15:0]) + cols);
        end
      end
      OP_GATHER: begin
        rows = 32'(c.w3[15:0]); cols = 32'(c.w3[31:16]);
        if (rows != 0 && cols != 0) begin
          f.rd[0] = mk(SP_TMEM, c.w1, (rows - 1) * c.w4 + cols);
          f.wr[0] = mk(SP_TMEM, c.w2, 32'(S - 1) * c.w6 + (rows - 1) * c.w5 + cols);
        end
      end
      OP_BAR: f.all = 1'b1;
      default: ;
    endcase
    return f;
  endfunction
endpackage
