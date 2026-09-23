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
// IMEM is a synchronous RAM (FPGA block RAM) of rows of IPR = D/32 instructions -- one DRAM
// chunk, so the slice's loader writes one row per cycle. It is read at the next pc, so the
// instruction at pc is available every cycle.
module otpu_seq
  import otpu_pkg::*;
#(
  parameter int IMEM_WORDS = 1 << 16,
  parameter int SID        = 0,
  parameter int S          = 1,
  parameter int D          = 32,
  parameter int WIN        = 16
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
  // IMEM write port (the loader; used while the slice is held in reset)
  input  logic                im_we,
  input  logic [31:0]         im_row,
  input  logic [D*8-1:0]      im_data
);
  localparam int SW = $clog2(WIN);
  localparam int IPR = D / 32;                   // instructions per IMEM row
  localparam int NROW = IMEM_WORDS / (8 * IPR);
  localparam int RW = $clog2(NROW);
  initial if (D % 32 != 0) $fatal(1, "otpu_seq: D must be a multiple of 32");
  logic [D*8-1:0] imem [NROW];
  logic [D*8-1:0] irow;
  logic [31:0]    isel;

  logic [31:0] pc, pc_n, cyc;
  logic [31:0] R [16];
  logic        stopping;         // HALT fetched: wait for the window to drain

  // loop stack
  logic [2:0]  sp;
  logic [31:0] stk_start [4];
  logic [31:0] stk_end   [4];
  logic [31:0] stk_rem   [4];
  // LOOP takes two cycles: the first registers the trip count and body length (pc holds), the
  // second enters or skips the loop from registers -- keeps the add off the fetch-address path
  logic        lp;
  logic [31:0] lp_cnt, lp_len;

  // window
  logic [WIN-1:0]  sv;                       // slot valid (dispatched, not completed)
  logic [WIN-1:0]  older [WIN];              // older[i][j]: slot j was dispatched before slot i
  int              sunit [WIN];
  logic [WIN-1:0]  sstarted, sready;
  logic [WIN-1:0]  sdep [WIN];               // sdep[i][j]: slot i waits for slot j
  logic [WIN-1:0]  sdepd [WIN];              // the DRAM part of sdep (MXU stream start)
  logic [WIN-1:0]  srel;                     // MXU: released
  logic [SW:0]     uq_r;                     // MXU: next started slot to release
  cmd_t            scmd [WIN];
  fp_t             sfp  [WIN];
  logic [31:0]     srdy_c [WIN];
  // per-unit queues of started slot ids, in start (= completion) order
  logic [SW-1:0]   uq [NUNITS][WIN];
  logic [SW:0]     uq_h [NUNITS], uq_t [NUNITS];

  // ---- fetch: the row holding pc_n is read at the clock edge that makes it pc
  always_ff @(posedge clk) begin
    if (im_we) imem[RW'(im_row)] <= im_data;
    irow <= imem[RW'(pc_n / IPR)];
    isel <= pc_n % IPR;
  end
  logic [31:0] iw [8];
  always_comb for (int k = 0; k < 8; k++) iw[k] = irow[32 * (8 * (isel % IPR) + k) +: 32];
  wire [7:0] op    = iw[0][7:0];
  wire [3:0] ra    = iw[0][11:8];
  wire [3:0] rb    = iw[0][15:12];
  wire [3:0] rc    = iw[0][19:16];
  wire [3:0] rd    = iw[0][23:20];
  wire [7:0] flags = iw[0][31:24];

  function automatic logic [31:0] rv(input logic [3:0] r);
    return (r == 0) ? 32'd0 : R[r];
  endfunction

  cmd_t dcmd;
  always_comb begin
    dcmd.op = op; dcmd.flags = flags;
    dcmd.w1 = iw[1] + rv(ra);
    dcmd.w2 = iw[2] + rv(rb);
    dcmd.w3 = iw[3] + rv(rc);
    dcmd.w4 = iw[4]; dcmd.w5 = iw[5]; dcmd.w6 = iw[6]; dcmd.w7 = iw[7];
  end
  int dunit;
  always_comb dunit = unit_of(op);

  // ---- dispatch pipeline: R (the instruction at pc: registers resolved) -> P (footprint
  // partial products) -> S (products) -> Q (footprint ranges) -> C (dependencies on the window,
  // slot allocation). In order; a stage holds while the next cannot take its instruction (C:
  // the window is full). The stages stream, so they add latency only at the program start.
  logic        p_v, s_v, q_v, c_v;
  cmd_t        p_cmd, s_cmd, q_cmd, c_cmd;
  int          p_unit, s_unit, q_unit, c_unit;
  logic [31:0] p_pc, s_pc, q_pc, c_pc;
  fpp_t        s_pp;
  fpm_t        q_pr;
  fp_t         c_fp;
  logic        have_free;
  wire         c_go = c_v && have_free;
  wire         q_adv = !c_v || c_go;            // Q hands its instruction to C
  wire         s_adv = !q_v || q_adv;           // S hands its instruction to Q
  wire         p_adv = !s_v || s_adv;           // P hands its instruction to S
  wire         r_take = !p_v || p_adv;          // P can take the instruction at R
  wire         pipe_empty = !p_v && !s_v && !q_v && !c_v;

  // ---- completions this cycle
  logic [WIN-1:0] fin;
  always_comb begin
    fin = '0;
    for (int u = 0; u < NUNITS; u++)
      if (udone[u]) fin[uq[u][uq_h[u][SW-1:0]]] = 1'b1;
  end

  // ---- free slot and the new instruction's dependencies
  logic [SW-1:0]   free_slot;
  logic [WIN-1:0]  ndep, ndepd;
  always_comb begin
    have_free = 1'b0; free_slot = '0;
    for (int i = WIN - 1; i >= 0; i--)
      if (!sv[i]) begin have_free = 1'b1; free_slot = SW'(i); end
    for (int i = 0; i < WIN; i++) begin
      ndep[i] = sv[i] && !fin[i] && conflict(c_fp, sfp[i]);
      ndepd[i] = sv[i] && !fin[i] && conflict_dram(c_fp, sfp[i]);
    end
  end

  // ---- loop-end handling for the instruction at pc
  logic        at_end;
  logic [31:0] next_pc;
  always_comb begin
    at_end  = (sp != 0) && (stk_end[sp-1] == pc);
    next_pc = pc + 1;
    if (at_end && stk_rem[sp-1] > 1) next_pc = stk_start[sp-1];
  end

  task automatic advance();
    if (at_end) begin
      if (stk_rem[sp-1] > 1) stk_rem[sp-1] <= stk_rem[sp-1] - 1;
      else sp <= sp - 1;
    end
  endtask

  // ---- the next pc (mirrors the fetch/dispatch below)
  always_comb begin
    pc_n = pc;
    if (rst) pc_n = '0;
    else if (!stopping && !halted)
      case (op)
        OP_NOP, OP_LI, OP_ADDI: pc_n = next_pc;
        OP_HALT: ;
        OP_LOOP: if (lp) pc_n = (lp_cnt == 0) ? pc + 1 + lp_len : pc + 1;
        default: if (dunit >= 0 && r_take) pc_n = next_pc;
      endcase
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
        same_started[u][i] = sv[i] && sstarted[i] && sunit[i] == u && u != U_VPU;
  end

  // ---- per-unit start: the oldest ready (all dependencies completed) instruction
  logic [NUNITS-1:0] can_start;
  logic [SW-1:0]     start_slot [NUNITS];
  // (the oldest of a set: the member none of whose older slots is in the set -- an age matrix
  // instead of comparing dispatch numbers, so the choice is parallel logic)
  always_comb begin
    for (int u = 0; u < NUNITS; u++) begin
      logic found;
      logic [WIN-1:0] cand;
      found = 1'b0; start_slot[u] = '0;
      // collectives pair up across slices, so the collective unit stays strictly in order:
      // it only considers its oldest instruction
      for (int i = 0; i < WIN; i++)
        cand[i] = sv[i] && !sstarted[i] && sunit[i] == u &&
                  ((sdep[i] & ~same_started[u]) == '0 || u == U_COLL || u == U_MXU);
      for (int i = 0; i < WIN; i++)
        if (cand[i] && (older[i] & cand) == '0) begin
          found = 1'b1;
          start_slot[u] = SW'(i);
        end
      if (u == U_MXU)
        can_start[u] = found && (sdepd[start_slot[u]] & ~same_started[u]) == '0 && urdy[u] &&
                       !ustart[u];
      else
        can_start[u] = found && (sdep[start_slot[u]] & ~same_started[u]) == '0 && urdy[u] &&
                       !ustart[u];
    end
  end


  // MXU release: the oldest started, unreleased MM once all its dependencies are clear
  wire [SW-1:0] rel_slot = uq[U_MXU][uq_r[SW-1:0]];
  wire          can_rel  = (uq_r != uq_t[U_MXU]) && (sdep[rel_slot] & ~same_started[U_MXU]) == '0;

`ifndef SYNTHESIS
  bit trace;
  initial trace = $test$plusargs("trace");
`endif

  always_ff @(posedge clk) begin
    ustart <= '0;
    urel <= 1'b0;
    if (rst) begin
      uq_r <= '0;
      pc <= '0; sp <= '0; cyc <= '0;
      halted <= 1'b0; error <= 1'b0; stopping <= 1'b0;
      p_v <= 1'b0; s_v <= 1'b0; q_v <= 1'b0; c_v <= 1'b0; lp <= 1'b0;
      icount <= '0;
      sv <= '0; sstarted <= '0; sready <= '0;
      for (int i = 0; i < 16; i++) R[i] <= '0;
      for (int u = 0; u < NUNITS; u++) begin uq_h[u] <= '0; uq_t[u] <= '0; end
      for (int i = 0; i < WIN; i++) sdep[i] <= '0;
    end else begin
      cyc <= cyc + 1;
      pc <= pc_n;
      // completions
      if (can_rel) begin
        urel <= 1'b1;
        uq_r <= uq_r + 1;
`ifndef SYNTHESIS
        if (trace) $display("T%0d G c=%0d s=%0d", SID, cyc, rel_slot);
`endif
      end
      for (int i = 0; i < WIN; i++) begin
        sdep[i] <= sdep[i] & ~fin;
        sdepd[i] <= sdepd[i] & ~fin;
        if (fin[i]) begin
          sv[i] <= 1'b0;
`ifndef SYNTHESIS
          if (trace) $display("T%0d E c=%0d s=%0d", SID, cyc, i);
`endif
        end
      end
      for (int u = 0; u < NUNITS; u++) if (udone[u]) uq_h[u] <= uq_h[u] + 1;
      // readiness (for the profiler: when did the dependencies clear)
      for (int i = 0; i < WIN; i++)
        if (sv[i] && !sready[i] && sdep[i] == '0) begin
          sready[i] <= 1'b1;
          srdy_c[i] <= cyc;
        end
      // starts
      for (int u = 0; u < NUNITS; u++) begin
        if (can_start[u]) begin
          ustart[u] <= 1'b1;
          ucmd[u] <= scmd[start_slot[u]];
          sstarted[start_slot[u]] <= 1'b1;
          uq[u][uq_t[u][SW-1:0]] <= start_slot[u];
          uq_t[u] <= uq_t[u] + 1;
`ifndef SYNTHESIS
          if (trace) $display("T%0d S c=%0d s=%0d u=%0d r=%0d", SID, cyc, start_slot[u], u,
                              sready[start_slot[u]] ? srdy_c[start_slot[u]] : cyc);
`endif
        end
      end
      // ---- C: dispatch into the window
      if (c_go) begin
        icount <= icount + 1;
        sv[free_slot] <= 1'b1;
        older[free_slot] <= sv;           // every slot in the window is older
        for (int i = 0; i < WIN; i++) older[i][free_slot] <= 1'b0;
        sunit[free_slot] <= c_unit;
        sstarted[free_slot] <= 1'b0;
        sready[free_slot] <= 1'b0;
        sdep[free_slot] <= ndep;
        sdepd[free_slot] <= ndepd;
        scmd[free_slot] <= c_cmd;
        sfp[free_slot] <= c_fp;
`ifndef SYNTHESIS
        if (trace) $display("T%0d D c=%0d s=%0d pc=%0d op=%02h w1=%08h w2=%08h w3=%08h",
                            SID, cyc, free_slot, c_pc, c_cmd.op, c_cmd.w1, c_cmd.w2, c_cmd.w3);
`endif
      end
      // ---- Q: footprint ranges
      if (q_adv) begin
        c_v <= q_v;
        if (q_v) begin
          c_cmd <= q_cmd; c_unit <= q_unit; c_pc <= q_pc;
          c_fp <= fp_ranges(q_cmd, q_pr, D, S);
        end
      end
      // ---- S: products
      if (s_adv) begin
        q_v <= s_v;
        if (s_v) begin
          q_cmd <= s_cmd; q_unit <= s_unit; q_pc <= s_pc;
          q_pr <= fp_sum(s_pp);
        end
      end
      // ---- P: partial products
      if (p_adv) begin
        s_v <= p_v;
        if (p_v) begin
          s_cmd <= p_cmd; s_unit <= p_unit; s_pc <= p_pc;
          s_pp <= fp_part(fp_ops(p_cmd, D, S));
        end
      end
      if (r_take) p_v <= 1'b0;
      // ---- R: fetch / execute control / hand over
      if (stopping) begin
        if (sv == '0 && pipe_empty) halted <= 1'b1;
      end else if (!halted) begin
        case (op)
          OP_NOP: begin icount <= icount + 1; advance(); end
          OP_HALT: begin icount <= icount + 1; stopping <= 1'b1; end
          OP_LI: begin
            icount <= icount + 1;
            if (rd != 0) R[rd] <= iw[1];
            advance();
          end
          OP_ADDI: begin
            icount <= icount + 1;
            if (rd != 0) R[rd] <= rv(ra) + iw[1];
            advance();
          end
          OP_LOOP: begin
            if (!lp) begin
              lp <= 1'b1;
              lp_cnt <= rv(ra) + iw[2];
              lp_len <= iw[1];
            end else begin
              lp <= 1'b0;
              icount <= icount + 1;
              if (lp_cnt != 0) begin
                stk_start[sp] <= pc + 1;
                stk_end[sp]   <= pc + lp_len;
                stk_rem[sp]   <= lp_cnt;
                sp <= sp + 1;
              end
            end
          end
          default: begin
            if (dunit < 0) begin
              error <= 1'b1;
              halted <= 1'b1;
            end else if (r_take) begin
              p_v <= 1'b1;
              p_cmd <= dcmd; p_unit <= dunit; p_pc <= pc;
              advance();
            end
          end
        endcase
      end
    end
  end

`ifndef SYNTHESIS
  string dir;
  logic [31:0] init_w [IMEM_WORDS];
  initial begin
    for (int i = 0; i < IMEM_WORDS; i++) init_w[i] = '0;
    if ($value$plusargs("dir=%s", dir) && !$test$plusargs("boot"))
      $readmemh($sformatf("%s/prog_%0d.hex", dir, SID), init_w);
    for (int r = 0; r < NROW; r++)
      for (int w = 0; w < 8 * IPR; w++) imem[r][32 * w +: 32] = init_w[r * 8 * IPR + w];
  end
`endif
endmodule
