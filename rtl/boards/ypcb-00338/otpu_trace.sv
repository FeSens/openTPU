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
//       count mod DEPTH; with stop, recording ends once DEPTH records are written
// Reading: rdata is the record at raddr, two cycles after raddr (the block RAM's output register).
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
  output logic [31:0] count,          // records written since the clear (saturating)
  output logic [31:0] drop,           // events lost to a full queue (saturating)
  output logic        busy            // events taken and not yet written (or dropped)
);
  localparam int LD = $clog2(DEPTH);
  localparam int QW = $clog2(QD);
  initial begin
    if (DEPTH != (1 << LD) || QD != (1 << QW)) $fatal(1, "otpu_trace: DEPTH, QD: powers of two");
    if (WIN > 16) $fatal(1, "otpu_trace: slot fields are 4 bits (WIN <= 16)");
  end
  localparam logic [3:0] T_D = 1, T_S = 2, T_G = 3, T_E = 4, T_U = 5, T_P = 6, T_Q = 7, T_H = 8,
                         T_O = 9, T_R = 10, T_V = 11;

  wire stopped = stop && (count[31:LD] != '0);        // count >= DEPTH

  // ------------------------------------------------------------------ S0: input register
  perf_t      ev;
  logic [5:0] ev_n;                                    // events (trace lines) in ev
  always_ff @(posedge clk) begin
    ev <= pf;
    ev_n <= 6'(pf.sq.d) + 6'($countones(pf.sq.s)) + 6'(pf.sq.g) + 6'($countones(pf.sq.e)) +
            6'(pf.u_mxu) + 6'(pf.u_q) + 6'(pf.u_vpu) + 6'(pf.h) + 6'({pf.w, 1'b0});
    if (rst || clear || !en || stopped) begin
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
  } bnd_t;

  bnd_t nb;
  always_comb begin
    nb.cyc = ev.sq.cyc;
    nb.g = ev.sq.g;
    nb.g_slot = ev.sq.g_slot[3:0];
    nb.e = ev.sq.e[15:0];
    nb.s = ev.sq.s;
    for (int u = 0; u < NUNITS; u++) begin
      logic [31:0] dt;
      dt = ev.sq.cyc - ev.sq.s_rdy[u];
      nb.s_slot[u] = ev.sq.s_slot[u][3:0];
      nb.s_x[u] = dt >= 32'hFFFF;
      nb.s_dt[u] = nb.s_x[u] ? 16'hFFFF : dt[15:0];
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
  end

  (* ram_style = "distributed" *) bnd_t qm [QD];
  logic [QW-1:0] q_t, q_hd;
  logic [QW:0]   q_n;
  wire push = ev_any && q_n < (QW+1)'(QD);
  always_ff @(posedge clk) if (push) qm[q_t] <= nb;
  bnd_t qh;
  assign qh = qm[q_hd];

  // ------------------------------------------------------------------ U: unpacker
  // a bundle's items, in record order
  localparam int I_G = 0, I_E = 1, I_S = 17, I_D = 27, I_U1 = 30, I_U2 = 35, I_U3 = 37,
                 I_H = 39, I_P = 44, I_Q = 54, NI = 61;
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
  wire   last = (pend & (pend - 1'b1)) == '0;          // at most one item left
  wire   load = last && q_n != 0;
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
  (* ram_style = "block" *) logic [63:0] mem [DEPTH];
  logic          r_v;
  logic [63:0]   r_rec;
  logic [LD-1:0] wp;
  wire           wr = r_v && !stopped;
  always_ff @(posedge clk) if (wr) mem[wp] <= r_rec;
  logic [63:0] rd_q;
  always_ff @(posedge clk) begin
    rd_q <= mem[raddr[LD-1:0]];
    rdata <= rd_q;
  end

  assign busy = ev_any || q_n != 0 || pend != '0 || r_v;

  always_ff @(posedge clk) begin
    if (rst || clear) begin
      q_t <= '0; q_hd <= '0; q_n <= '0;
      pend <= '0; r_v <= 1'b0;
      wp <= '0; count <= '0; drop <= '0;
    end else begin
      if (push) q_t <= q_t + 1'b1;
      if (load) begin
        cur <= qh;
        pend <= items(qh);
        q_hd <= q_hd + 1'b1;
      end else begin
        pend <= pend & ~first;
      end
      q_n <= q_n + (QW+1)'(push) - (QW+1)'(load);
      if (ev_any && !push) drop <= (drop > ~32'(ev_n)) ? '1 : drop + 32'(ev_n);
      r_v <= pend != '0;
      r_rec <= rec;
      if (wr) begin
        wp <= wp + 1'b1;
        if (count != '1) count <= count + 1;
      end
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst && pf.sq.d && pf.sq.d_pc[31:16] != '0)
      $fatal(1, "otpu_trace: pc %0d does not fit the D record's 16 bits", pf.sq.d_pc);
`endif
endmodule
