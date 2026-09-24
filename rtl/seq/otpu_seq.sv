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
// chunk, so the slice's loader writes one row per cycle. It is read one instruction ahead, at
// fa: the exact pc that R moves to next (computed from flip-flops only). When R moves on, the
// instruction at fa is registered into ir, so no R-stage path starts at the RAM output. This
// costs one cycle at the program start (ir is empty after reset).
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
  logic [D*8-1:0] irow;           // the RAM output: the row holding fa
  logic [255:0]   ir;             // the instruction at pc
  logic           ir_v;           // ir is valid (0 only in the first cycle after reset)
  logic [31:0]    fa, fa_d;       // the pc R moves to next; its next value (the RAM address)
  logic           ld;             // R moves on to the instruction at fa

  logic [31:0] pc, cyc;
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

  // The footprint as the window keeps it: an MM's ASCALE TMEM range (fp_t rd[3]) never comes
  // with its streamed-scale DRAM range (rd[1]; ASCALE needs UNIT, docs/isa.md), so it moves
  // into rd[1] -- 16 range pairs per slot instead of 20. Should both be set anyway (an invalid
  // MM), the instruction conflicts with everything instead: a superset of its dependencies.
  typedef struct packed {
    logic            all;
    rng_t [2:0]      rd;
    rng_t [1:0]      wr;
  } fp5_t;

  function automatic fp5_t fp_pack(input fp_t f);
    fp5_t p;
    p.all = f.all; p.rd = f.rd[2:0]; p.wr = f.wr;
    if (f.rd[3].v) begin
      if (f.rd[1].v) p.all = 1'b1;
      else p.rd[1] = f.rd[3];
    end
    return p;
  endfunction

  // {conflict, conflict_dram} (otpu_pkg) on packed footprints, each range pair tested once: a
  // pair can only overlap within one space, so the DRAM part is the overlaps of DRAM ranges
  function automatic logic [1:0] conf5(input fp5_t n, input fp5_t e);
    logic any, dram;
    any = n.all || e.all; dram = any;
    for (int i = 0; i < 2; i++) begin
      for (int j = 0; j < 3; j++) begin
        if (ov(n.wr[i], e.rd[j])) begin any = 1'b1; dram |= n.wr[i].sp == SP_DRAM; end
        if (ov(e.wr[i], n.rd[j])) begin any = 1'b1; dram |= e.wr[i].sp == SP_DRAM; end
      end
      for (int j = 0; j < 2; j++)
        if (ov(n.wr[i], e.wr[j])) begin any = 1'b1; dram |= n.wr[i].sp == SP_DRAM; end
    end
    return {any, dram};
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
  cmd_t            scmd [WIN];
  fp5_t            sfp  [WIN];
  logic [31:0]     srdy_c [WIN];
  // per-unit queues of started slot ids, in start (= completion) order
  logic [SW-1:0]   uq [NUNITS][WIN];
  logic [SW:0]     uq_h [NUNITS], uq_t [NUNITS];

  // ---- fetch: the row holding fa_d is read at the clock edge that makes it fa; ir takes the
  // instruction at fa at the edge that makes fa pc
  always_ff @(posedge clk) begin
    if (im_we) imem[RW'(im_row)] <= im_data;
    irow <= imem[RW'(fa_d / IPR)];
  end
  always_ff @(posedge clk) begin
    fa <= fa_d;
    if (rst) ir_v <= 1'b0;
    else if (ld) begin
      ir <= irow[256 * (fa % IPR) +: 256];
      ir_v <= 1'b1;
    end
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
  logic [2:0]  p_unit, s_unit, q_unit, c_unit;
  logic [31:0] p_pc, s_pc, q_pc, c_pc;
  fpp_t        s_pp;
  fpm_t        q_pr;
  fp5_t        c_fp;
  logic        have_free;
  wire         c_go = c_v && have_free;
  wire         q_adv = !c_v || c_go;            // Q hands its instruction to C
  wire         s_adv = !q_v || q_adv;           // S hands its instruction to Q
  wire         p_adv = !s_v || s_adv;           // P hands its instruction to S
  wire         r_take = !p_v || p_adv;          // P can take the instruction at R
  wire         pipe_empty = !p_v && !s_v && !q_v && !c_v;
  // R retires a control instruction this cycle (NOP, HALT, LI, ADDI, a LOOP's second cycle)
  logic        r_ret;

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
      logic [1:0] cf;
      cf = conf5(c_fp, sfp[i]);
      ndep[i] = sv[i] && !fin[i] && cf[1];
      ndepd[i] = sv[i] && !fin[i] && cf[0];
    end
  end

  // ---- loop-end handling for the instruction at pc
  logic        at_end;
  always_comb at_end = (sp != 0) && (stk_end[sp-1] == pc);

  task automatic advance();
    if (at_end) begin
      if (stk_rem[sp-1] > 1) stk_rem[sp-1] <= stk_rem[sp-1] - 1;
      else sp <= sp - 1;
    end
  endtask

  always_comb
    r_ret = ir_v && !stopping && !halted &&
            (op == OP_NOP || op == OP_HALT || op == OP_LI || op == OP_ADDI ||
             (op == OP_LOOP && lp));

  // ---- the next fetch address (mirrors the fetch/dispatch below). fa is the successor of pc
  // under the loop stack as it stands while R holds pc; when R moves on, fa_d is the successor
  // of fa under the stack as R's instruction leaves it:
  //   A: a LOOP's second cycle pushes (fa == pc + 1 then) -- a one-instruction body repeats
  //   B: advance() at a loop end decrements the top      -- repeats if the top keeps rem > 1
  //   C: advance() at a loop end pops                    -- the next entry decides
  //   D: the stack is unchanged (also the load after reset and a skipped LOOP)
  // A LOOP's first cycle picks the enter/skip target a cycle ahead (pc holds, the RAM reads it).
  logic        adv;              // R's instruction retires or is handed over this cycle
  logic        hit1, hit2;       // fa ends the loop on top of the stack / the one below it
  logic        hit;              // fa's successor is a loop start ...
  logic [31:0] hit_pc;           // ... this one
  logic        lp_z;             // the LOOP's trip count (R[ra] + iw[2]) is 0
  logic [31:0] lp_a;
  logic [31:0] succ;
  always_comb begin
    case (op)
      OP_NOP, OP_LI, OP_ADDI: adv = 1'b1;
      OP_HALT: adv = 1'b0;
      OP_LOOP: adv = lp;
      default: adv = dunit >= 0 && r_take;
    endcase
    ld = !ir_v || (!stopping && !halted && adv);
    hit1 = (sp != 0) && (stk_end[sp-1] == fa);
    hit2 = (sp >= 2) && (stk_end[sp-2] == fa);
    hit = hit1 && stk_rem[sp-1] > 1;                                          // D
    hit_pc = stk_start[sp-1];
    if (ir_v && op == OP_LOOP) begin
      if (lp_cnt != 0) begin hit = lp_len == 1 && lp_cnt > 1; hit_pc = fa; end  // A
    end else if (ir_v && at_end) begin
      if (stk_rem[sp-1] > 1) hit = hit1 && stk_rem[sp-1] > 2;                // B
      else begin hit = hit2 && stk_rem[sp-2] > 1; hit_pc = stk_start[sp-2]; end // C
    end
    succ = hit ? hit_pc : fa + 1;
    // a + b == 0 without the carry chain: every sum bit is 0 iff a ^ b is the carry, (a | b) << 1
    lp_a = rv(ra);
    lp_z = (lp_a ^ iw[2]) == ((lp_a | iw[2]) << 1);
    if (rst) fa_d = '0;
    else if (ir_v && op == OP_LOOP && !lp && !stopping && !halted)
      fa_d = lp_z ? pc + 1 + iw[1] : pc + 1;
    else if (ld) fa_d = succ;
    else fa_d = fa;
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
  // in parallel with the selection and AND-ORed with sel -- no slot id on the can_start path
  always_comb begin
    for (int i = 0; i < WIN; i++) begin
      mrdy[i] = (sdepd[i] & ~same_started[U_MXU]) == '0;
      crdy[i] = (sdep[i] & ~same_started[U_COLL]) == '0;
    end
    for (int u = 0; u < NUNITS; u++) begin
      logic [WIN-1:0] cand;
      start_slot[u] = '0;
      // collectives pair up across slices, so the collective unit stays strictly in order:
      // it only considers its oldest instruction
      for (int i = 0; i < WIN; i++)
        cand[i] = sv[i] && !sstarted[i] && soh[i][u] &&
                  ((sdep[i] & ~same_started[u]) == '0 || u == U_COLL || u == U_MXU);
      for (int i = 0; i < WIN; i++) begin
        sel[u][i] = cand[i] && (older[i] & cand) == '0;
        if (sel[u][i]) start_slot[u] |= SW'(i);
      end
      cand_any[u] = |cand;
      if (u == U_MXU)
        can_start[u] = |(sel[u] & mrdy) && urdy[u] && !ustart[u];
      else if (u == U_COLL)
        can_start[u] = |(sel[u] & crdy) && urdy[u] && !ustart[u];
      else    // cand already requires readiness
        can_start[u] = cand_any[u] && urdy[u] && !ustart[u];
    end
  end


  // MXU release: the oldest started, unreleased MM once all its dependencies are clear
  wire [SW-1:0] rel_slot = uq[U_MXU][uq_r[SW-1:0]];
  wire          can_rel  = (uq_r != uq_t[U_MXU]) && (sdep[rel_slot] & ~same_started[U_MXU]) == '0;

  // Each unit's command is read from the window at the slot it last started: a slot's scmd is
  // written only at dispatch and stays until the slot completes, so it holds the command for as
  // long as the unit reads it (at ustart; the collective until its udone). After completion the
  // slot may be reused and ucmd shows the new occupant -- no unit reads it then.
  logic [SW-1:0] ucs [NUNITS];
  always_comb for (int u = 0; u < NUNITS; u++) ucmd[u] = scmd[ucs[u]];

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
      if (ld) pc <= fa;
`ifndef SYNTHESIS
      // fa is the pc the one-instruction-per-cycle fetch would move to (the next_pc rule)
      if (ld && ir_v &&
          fa != ((op == OP_LOOP) ? ((lp_cnt == 0) ? pc + 1 + lp_len : pc + 1) :
                 (at_end && stk_rem[sp-1] > 1) ? stk_start[sp-1] : pc + 1))
        $fatal(1, "otpu_seq: fetch address %0d does not follow pc %0d", fa, pc);
`endif
      icount <= icount + 32'(r_ret) + 32'(c_go);
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
`ifndef SYNTHESIS
        // the age matrix orders every pair of candidates: exactly one oldest when any
        if (!$onehot0(sel[u]) || cand_any[u] != |sel[u])
          $fatal(1, "otpu_seq: unit %0d oldest-candidate select %b is not one-hot", u, sel[u]);
`endif
        for (int i = 0; i < WIN; i++)
          if (can_start[u] && sel[u][i]) sstarted[i] <= 1'b1;
        if (can_start[u]) begin
          ustart[u] <= 1'b1;
          ucs[u] <= start_slot[u];
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
        sv[free_slot] <= 1'b1;
        older[free_slot] <= sv;           // every slot in the window is older
        for (int i = 0; i < WIN; i++) older[i][free_slot] <= 1'b0;
        soh[free_slot] <= NUNITS'(1) << c_unit;
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
          c_fp <= fp_pack(fp_ranges(q_cmd, q_pr, D, S));
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
      end else if (!halted && ir_v) begin
        case (op)
          OP_NOP: advance();
          OP_HALT: stopping <= 1'b1;
          OP_LI: begin
            if (rd != 0) R[rd] <= iw[1];
            advance();
          end
          OP_ADDI: begin
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
              p_cmd <= dcmd; p_unit <= 3'(dunit); p_pc <= pc;
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
