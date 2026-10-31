// Hardware trace: the slice's trace events (otpu_pkg perf_t: the lines +trace prints in
// simulation, opentpu/profile.py) as 64-bit records in a block-RAM ring the host reads through
// the control registers (otpu_ctrl.sv; the record format is in docs/observability.md, the host
// decoder is opentpu/hwtrace.py).
//
//   S0  the events are registered (and dropped unless recording: en, and not stopped)
//   S1  a cycle with events becomes one bundle in a small queue (QD bundles, LUT RAM); a bundle
//       that finds the queue full is lost and its events (trace lines) are counted in drop
//   U   the unpacker turns the head bundle into its records, one per cycle, in the order the
//       simulator prints the lines: G, E (by slot), S (by unit; R after an S whose delay field
//       saturated), D + two O, U + V (MXU, QUANT, VPU), H + V, P + V, Q + V
//   W   the record is written to the ring (DEPTH records, simple dual-port block RAM) at
//       count mod DEPTH; with stop, recording ends once DEPTH records are written. count
//       saturates in its high bits only: past 2^32 - DEPTH records its low log2(DEPTH) bits keep
//       counting, so count mod DEPTH is the next record's slot (the host's oldest record) on any
//       run, and count stays >= DEPTH
// Reading: rdata is the record at raddr, three cycles after raddr (the block RAM's read and two
// output registers; otpu_ctrl's readout leaves them the time, below).
// busy: events are on their way to the ring; after the halt the host waits for it to clear
// (the events of the last cycles, up to QD bundles of up to 61 records, are still draining).
module otpu_trace
  import otpu_pkg::*;
#(
  parameter int DEPTH = 16384,        // records (a power of two)
  parameter int QD    = 32,           // bundles in the capture queue (a power of two)
  parameter int WIN   = 16            // the sequencer's window slots (<= 16: 4-bit slot fields)
) (
  input  logic        clk,
  input  logic        rst,
  input  perf_t       pf,
  input  logic        en,             // record (ENABLE and RUN)
  input  logic        stop,           // STOP_WHEN_FULL
  input  logic        clear,          // empty the ring, zero count and drop
  input  logic [31:0] raddr,
  output logic [63:0] rdata,
  output logic [31:0] count,          // records written since the clear (saturating; the low
                                      //   log2(DEPTH) bits always the write slot)
  output logic [31:0] drop,           // events lost to a full queue (saturating)
  output logic        busy            // events taken and not yet written (or dropped)
);
  localparam int LD = $clog2(DEPTH);
  localparam int QW = $clog2(QD);
  initial begin
    if (DEPTH != (1 << LD) || QD != (1 << QW)) $fatal(1, "otpu_trace: DEPTH, QD: powers of two");
    if (WIN > 16) $fatal(1, "otpu_trace: slot fields are 4 bits (WIN <= 16)");
  end
`ifndef SYNTHESIS
  // +trace_count0=<hex> (tests/test_observability.py): a clear starts count (and the write slot)
  // there instead of 0, to run a ring past 2^32 records in a short simulation
  logic [31:0] cnt0;
  initial if (!$value$plusargs("trace_count0=%h", cnt0)) cnt0 = '0;
`endif
  localparam logic [3:0] T_D = 1, T_S = 2, T_G = 3, T_E = 4, T_U = 5, T_P = 6, T_Q = 7, T_H = 8,
                         T_O = 9, T_R = 10, T_V = 11;

  wire stopped = stop && (count[31:LD] != '0);        // count >= DEPTH

  // ------------------------------------------------------------------ P: the slice's events,
  // registered as they come (with the enable): the sequencer's start events look up the started
  // slot's ready cycle combinationally (otpu_seq), a 0.27 ns path into S0 at 125.49 MHz
  perf_t pr;
  logic  pr_en, pr_any;
  always_ff @(posedge clk) begin
    pr <= pf;
    pr_en <= en;
    pr_any <= !rst && (pf.sq.d || (|pf.sq.s) || pf.sq.g || (|pf.sq.e) || pf.u_mxu || pf.u_q ||
                       pf.u_vpu || pf.h || pf.w);
  end

  // ------------------------------------------------------------------ S0: input register
  // ev_sx: the S records' R flags (cycle - ready >= 0xFFFF), from pr: S1's item count (bnd_t ni)
  // starts at flip-flops, not behind the subtracts
  perf_t      ev;
  logic [5:0] ev_n;                                    // events (trace lines) in ev
  logic [NUNITS-1:0] ev_sx;
  always_ff @(posedge clk) begin
    logic [5:0] n;
    n = 6'(pr.sq.d) + 6'(pr.sq.g) + 6'(pr.u_mxu) + 6'(pr.u_q) + 6'(pr.u_vpu) + 6'(pr.h) +
        6'({pr.w, 1'b0});
    for (int u = 0; u < NUNITS; u++) n = n + 6'(pr.sq.s[u]);
    for (int i = 0; i < 16; i++) n = n + 6'(pr.sq.e[i]);
    ev <= pr;
    ev_n <= n;
    for (int u = 0; u < NUNITS; u++) ev_sx[u] <= pr.sq.cyc - pr.sq.s_rdy[u] >= 32'hFFFF;
    if (rst || clear || !pr_en || stopped) begin
      ev.sq.d <= 1'b0; ev.sq.s <= '0; ev.sq.g <= 1'b0; ev.sq.e <= '0;
      ev.u_mxu <= 1'b0; ev.u_q <= 1'b0; ev.u_vpu <= 1'b0; ev.h <= 1'b0; ev.w <= 1'b0;
    end
  end
  wire ev_any = ev.sq.d || (|ev.sq.s) || ev.sq.g || (|ev.sq.e) || ev.u_mxu || ev.u_q ||
                ev.u_vpu || ev.h || ev.w;

  // ------------------------------------------------------------------ S1: bundle, queue
  typedef struct packed {
    logic [31:0]              cyc;
    logic                     g;
    logic [3:0]               g_slot;
    logic [15:0]              e;
    logic [NUNITS-1:0]        s;
    logic [NUNITS-1:0][3:0]   s_slot;
    logic [NUNITS-1:0][15:0]  s_dt;        // cycle - ready; 16'hFFFF: s_x, an R record follows
    logic [NUNITS-1:0]        s_x;
    logic [NUNITS-1:0][31:0]  s_rdy;
    logic                     d;
    logic [3:0]               d_slot;
    logic [15:0]              d_pc;
    logic [103:0]             d_ow;        // {op, w3, w2, w1}
    logic                     u_mxu;
    logic [3:0][31:0]         u_mxu_v;
    logic                     u_q;
    logic [31:0]              u_q_v;
    logic                     u_vpu;
    logic [31:0]              u_vpu_v;
    logic                     h;
    logic [NH-1:0][31:0]      h_v;
    logic                     w;
    logic [23:0]              w_n;
    logic [NP-1:0][31:0]      w_p;
    logic [NQ-1:0][31:0]      w_q;
    logic [5:0]               ni;          // its items (records): items()'s ones
  } bnd_t;

  // the item list's layout (U, below): a bundle's records per field
  localparam int I_G = 0, I_E = 1, I_S = 17, I_D = 27, I_U1 = 30, I_U2 = 35, I_U3 = 37,
                 I_H = 39, I_P = 44, I_Q = 54, NI = 61;

  bnd_t nb;
  always_comb begin
    nb.cyc = ev.sq.cyc;
    nb.g = ev.sq.g;
    nb.g_slot = ev.sq.g_slot[3:0];
    nb.e = ev.sq.e[15:0];
    nb.s = ev.sq.s;
    for (int u = 0; u < NUNITS; u++) begin
      logic [15:0] dt;
      dt = ev.sq.cyc[15:0] - ev.sq.s_rdy[u][15:0];
      nb.s_slot[u] = ev.sq.s_slot[u][3:0];
      nb.s_x[u] = ev_sx[u];
      nb.s_dt[u] = nb.s_x[u] ? 16'hFFFF : dt;
      nb.s_rdy[u] = ev.sq.s_rdy[u];
    end
    nb.d = ev.sq.d;
    nb.d_slot = ev.sq.d_slot[3:0];
    nb.d_pc = ev.sq.d_pc[15:0];
    nb.d_ow = {ev.sq.d_op, ev.sq.d_w3, ev.sq.d_w2, ev.sq.d_w1};
    nb.u_mxu = ev.u_mxu;
    nb.u_mxu_v = ev.u_mxu_v;
    nb.u_q = ev.u_q;
    nb.u_q_v = ev.u_q_frz;
    nb.u_vpu = ev.u_vpu;
    nb.u_vpu_v = ev.u_vpu_frz;
    nb.h = ev.h;
    nb.h_v = ev.h_v;
    nb.w = ev.w;
    nb.w_n = ev.w_n[23:0];
    nb.w_p = ev.w_p;
    nb.w_q = ev.w_q;
    // items(nb)'s ones, field by field
    nb.ni = 6'(nb.g) + 6'((I_U1 - I_D) * nb.d) + 6'((I_U2 - I_U1) * nb.u_mxu) +
            6'((I_U3 - I_U2) * nb.u_q) + 6'((I_H - I_U3) * nb.u_vpu) + 6'((I_P - I_H) * nb.h) +
            6'((NI - I_P) * nb.w);
    for (int i = 0; i < 16; i++) nb.ni = nb.ni + 6'(nb.e[i]);
    for (int u = 0; u < NUNITS; u++) nb.ni = nb.ni + 6'(nb.s[u]) + 6'(nb.s[u] && nb.s_x[u]);
  end

  (* ram_style = "distributed" *) bnd_t qm [QD];
  logic [QW-1:0] q_t, q_hd;
  logic [QW:0]   q_n;
  wire push = ev_any && q_n < (QW+1)'(QD);
  always_ff @(posedge clk) if (push) qm[q_t] <= nb;
  bnd_t qh;
  assign qh = qm[q_hd];

  // ------------------------------------------------------------------ U: unpacker
  // a bundle's items, in record order (I_*: above)
  function automatic logic [NI-1:0] items(input bnd_t b);
    logic [NI-1:0] m;
    m = '0;
    m[I_G] = b.g;
    for (int i = 0; i < 16; i++) m[I_E + i] = b.e[i];
    for (int u = 0; u < NUNITS; u++) begin
      m[I_S + 2 * u] = b.s[u];
      m[I_S + 2 * u + 1] = b.s[u] && b.s_x[u];
    end
    for (int i = 0; i < 3; i++) m[I_D + i] = b.d;
    for (int i = 0; i < 5; i++) m[I_U1 + i] = b.u_mxu;
    for (int i = 0; i < 2; i++) m[I_U2 + i] = b.u_q;
    for (int i = 0; i < 2; i++) m[I_U3 + i] = b.u_vpu;
    for (int i = 0; i < 1 + NH; i++) m[I_H + i] = b.h;
    for (int i = 0; i < 1 + NP; i++) m[I_P + i] = b.w;
    for (int i = 0; i < 1 + NQ; i++) m[I_Q + i] = b.w;
    return m;
  endfunction

  // a V record: a counter value
  function automatic logic [63:0] val(input logic [3:0] idx, input logic [31:0] v);
    return {T_V, idx, 24'd0, v};
  endfunction

  bnd_t          cur;                                  // the bundle being unpacked
  logic [NI-1:0] pend, first;                          // its items left; the next one
  logic [63:0]   irec [NI];
  logic [63:0]   rec;
  assign first = pend & (~pend + 1'b1);
  // load: the head bundle into cur once at most one item is left in pend (and the queue has one),
  // from a register (ld), computed a cycle ahead from pend's count (cnt: its ones; the bundle's
  // own, ni, when loaded) and the queue's, replicated over cur's ~1,000 enables (250 MHz, FK35:
  // pend -> pend - 1 == 0 -> cur's enables, 8 levels, 74% route, -0.31 ns)
  (* max_fanout = 64 *) logic ld;
  logic [5:0]    cnt;
  wire   load = ld;
  always_comb begin
    for (int i = 0; i < NI; i++) irec[i] = '0;
    irec[I_G] = {T_G, 4'd0, cur.g_slot, 20'd0, cur.cyc};
    for (int i = 0; i < 16; i++) irec[I_E + i] = {T_E, 4'd0, 4'(i), 20'd0, cur.cyc};
    for (int u = 0; u < NUNITS; u++) begin
      irec[I_S + 2 * u] = {T_S, 4'd0, cur.s_slot[u], 4'(u), cur.s_dt[u], cur.cyc};
      irec[I_S + 2 * u + 1] = {T_R, 28'd0, cur.s_rdy[u]};
    end
    irec[I_D] = {T_D, 4'd0, cur.d_slot, 4'd0, cur.d_pc, cur.cyc};
    irec[I_D + 1] = {T_O, 7'd0, 1'b0, cur.d_ow[51:0]};
    irec[I_D + 2] = {T_O, 7'd0, 1'b1, cur.d_ow[103:52]};
    irec[I_U1] = {T_U, 8'd0, 4'd1, 16'd0, cur.cyc};
    for (int i = 0; i < 4; i++) irec[I_U1 + 1 + i] = val(4'(i), cur.u_mxu_v[i]);
    irec[I_U2] = {T_U, 8'd0, 4'd2, 16'd0, cur.cyc};
    irec[I_U2 + 1] = val(4'd0, cur.u_q_v);
    irec[I_U3] = {T_U, 8'd0, 4'd3, 16'd0, cur.cyc};
    irec[I_U3 + 1] = val(4'd0, cur.u_vpu_v);
    irec[I_H] = {T_H, 28'd0, cur.cyc};
    for (int i = 0; i < NH; i++) irec[I_H + 1 + i] = val(4'(i), cur.h_v[i]);
    irec[I_P] = {T_P, 4'd0, cur.w_n, cur.cyc};
    for (int i = 0; i < NP; i++) irec[I_P + 1 + i] = val(4'(i), cur.w_p[i]);
    irec[I_Q] = {T_Q, 4'd0, cur.w_n, cur.cyc};
    for (int i = 0; i < NQ; i++) irec[I_Q + 1 + i] = val(4'(i), cur.w_q[i]);
    // the next item's record: per bit, an AND-OR over the items (a balanced tree, not a chain)
    for (int b = 0; b < 64; b++) begin
      logic [NI-1:0] col;
      for (int i = 0; i < NI; i++) col[i] = irec[i][b];
      rec[b] = |(first & col);
    end
  end

  // ------------------------------------------------------------------ W: the ring
  // the read and two output registers (rd_q, rd_p: the cascade's pipeline at 250 MHz, and rdata).
  // otpu_ctrl takes a record four cycles after TRACE_ADDR moves at the earliest (TRACE_HI's
  // increment, the rest of its read and the next read's address stage; a TRACE_ADDR write's
  // response through otpu_axil_iso first: checked there), so the third register costs the host
  // nothing.
  (* ram_style = "block" *) logic [63:0] mem [DEPTH];
  logic          r_v;
  logic [63:0]   r_rec;
  logic [LD-1:0] wp;
  wire           wr = r_v && !stopped;
  always_ff @(posedge clk) if (wr) mem[wp] <= r_rec;
  logic [63:0] rd_q, rd_p;
  always_ff @(posedge clk) begin
    rd_q  <= mem[raddr[LD-1:0]];
    rd_p  <= rd_q;
    rdata <= rd_p;
  end

  assign busy = (pr_any && pr_en) || ev_any || q_n != 0 || pend != '0 || r_v;

  // cur loads with ld alone (in a reset or clear too: pend is emptied then, and the next load
  // overwrites it), so its enables are ld's replicas
  always_ff @(posedge clk) if (load) cur <= qh;
  always_ff @(posedge clk) begin
    if (rst || clear) begin
      q_t <= '0; q_hd <= '0; q_n <= '0;
      pend <= '0; r_v <= 1'b0;
      cnt <= '0; ld <= 1'b0;
`ifdef SYNTHESIS
      wp <= '0; count <= '0;
`else
      wp <= cnt0[LD-1:0]; count <= cnt0;
`endif
      drop <= '0;
    end else begin
      if (push) q_t <= q_t + 1'b1;
      if (load) begin
        pend <= items(qh);
        cnt <= qh.ni;
        q_hd <= q_hd + 1'b1;
      end else begin
        pend <= pend & ~first;
        if (cnt != '0) cnt <= cnt - 1'b1;
      end
      // the next cycle's load: at most one item left then (the loaded bundle's, or two now) and
      // a bundle in the queue then (one pushed, or one more now than this cycle's load takes)
      ld <= (load ? qh.ni <= 6'd1 : cnt <= 6'd2) &&
            (push || q_n > (QW+1)'(load));
      q_n <= q_n + (QW+1)'(push) - (QW+1)'(load);
      if (ev_any && !push) drop <= (drop > ~32'(ev_n)) ? '1 : drop + 32'(ev_n);
      r_v <= pend != '0;
      r_rec <= rec;
      if (wr) begin
        wp <= wp + 1'b1;
        count[LD-1:0] <= count[LD-1:0] + 1'b1;
        if (count[LD-1:0] == '1 && count[31:LD] != '1) count[31:LD] <= count[31:LD] + 1'b1;
      end
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && pf.sq.d && pf.sq.d_pc[31:16] != '0)
      $fatal(1, "otpu_trace: pc %0d does not fit the D record's 16 bits", pf.sq.d_pc);
  // the write slot is count mod DEPTH
  always_ff @(posedge clk)
    if (!rst && !clear && wp != count[LD-1:0])
      $fatal(1, "otpu_trace: write slot %0d, count %0d", wp, count);
  // ld is the load as pend and the queue stand: at most one item left, a bundle queued; cnt is
  // pend's ones
  always_ff @(posedge clk)
    if (!rst && !clear && (cnt != 6'($countones(pend)) ||
                           ld != (((pend & (pend - 1'b1)) == '0) && q_n != 0)))
      $fatal(1, "otpu_trace: load %0d with pend %h (count %0d), queue %0d", ld, pend, cnt, q_n);
`endif
endmodule
