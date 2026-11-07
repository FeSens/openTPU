// Collective unit: BAR and GATHER across S slices. When every slice is waiting on the same
// collective, GATHER streams each slice's segment, up to LANES consecutive words per cycle, and
// broadcasts it into the TMEM of all slices at dst + s*seg + r*drs + c; then all slices are
// released together. `gnt` is the AND of every slice's TMEM grant; without it the unit holds.
// RQ (TMEM's input stage, release r2, docs/tmem_nocopy.md 7): the reads go to the slices from a
// request register (stage a), a cycle after they form, as every unit's do, and a read's data is
// taken one granted cycle later than without it. For the slices' TMEMs on the unit's own ports
// (otpu_top; otpu_board at S = 1); otpu_board's hub at S > 1 takes RQ 0 (the ports present).
// UG (otpu_board's hub at S > 1): r_addr, w_addr and w_data are not gated by their enables (the
// hub's crossing registers take them as they are and its ports use them only with a read or a
// write), so they come from the unit's registers alone (250 MHz, FK35: u_coll's state -> the
// enables gating 2 x 520 address and data bits into g_x[].u_dn_tx, synthesized as their reset
// pins, 1 level, 95% route, -0.61 ns). UG 0: zero without their enable, for the TMEM ports.
// The issue stage's next position is decided from flags kept with it (c_l, r_l: the row's last
// chunk, the slice's last row; ssel: the slice), the next read address from adders beside each
// other, not the compares and two chained adders behind s (250 MHz, FK35: s -> ra, 11 levels,
// -0.43 ns); a read's position past the issue stage is just whether it is the last (plast).
module otpu_coll
  import otpu_pkg::*;
#(
  parameter int S     = 1,
  parameter int LANES = 8,
  parameter bit RQ    = 1'b1,
  parameter bit UG    = 1'b0
) (
  input  logic                            clk,
  input  logic                            rst,
  input  logic [S-1:0]                    req,
  input  cmd_t                            cmds [S],
  input  logic                            gnt,
  output logic                            ack,
  output logic [S-1:0][LANES-1:0]         r_en,
  output logic [S-1:0][LANES-1:0][31:0]   r_addr,
  input  logic [S-1:0][LANES-1:0][31:0]   r_data,
  output logic [LANES-1:0]                w_en,
  output logic [LANES-1:0][31:0]          w_addr,
  output logic [LANES-1:0][31:0]          w_data
);
  typedef enum logic [1:0] {C_IDLE, C_RUN, C_ACK, C_COOL} st_t;
  st_t st;
  logic [31:0] dst, drs, seg;
  logic [15:0] rows, cols;
  logic [31:0] s, r, c;
  logic        pv, issuing;
  // the issue stage's flags: c_l c + LANES >= cols (the row's last chunk), r_l r + 1 == rows (the
  // slice's last row; s's last is ssel[S-1]), and their values at c = 0 / r = 0 (cl0, rl0) and rm's
  // (rm0); stage p's (and a's) slice one-hot (psel, asel) and last-read flag (plast, alast)
  logic             c_l, r_l, cl0, rl0, plast, alast;
  logic [LANES-1:0] rm0;
  logic [S-1:0]     psel;
  // row base addresses kept by adding the strides (no multipliers):
  // rrow = w1[s] + r*w4[s], wrow = dst + s*seg + r*drs, wseg = dst + s*seg
  logic [31:0] rrow, wrow, wseg;
  // the read stage's address (rrow + c) and lane mask (c + l < cols), registered with c and
  // rrow: the TMEM read address pins take a flip-flop, not an adder (c -> TMEM block RAM
  // address was a 0.24 ns path at 125.49 MHz)
  logic [31:0]      ra;
  logic [LANES-1:0] rm;
  // the read requests from flip-flops: rd_on is st == C_RUN && issuing, ssel the one-hot of s
  // (a 32-bit compare of s and the enable gating were in front of the TMEM read address: the
  // build's u_coll s -> TMEM block RAM address path)
  logic             rd_on;
  logic [S-1:0]     ssel;
  // the write stage's address base and lane mask, registered with pcl (they feed the TMEM
  // arbiter, so no adder or compare sits in front of it)
  logic [31:0]      pwa;
  logic [LANES-1:0] pm;
  // write stage q: stage p's address, mask and TMEM read data, registered (no path from the
  // TMEM block RAMs to their write port in one cycle); qlast: it holds the command's last write.
  // The lanes' addresses are registered each (qwl[l] = pwa + l): the arbiter's bank masks start
  // at flip-flops, not at a carry chain (133.33 MHz, e698dcd: qwa -> qwa + l -> the collective's
  // bank mask -> every lower unit's grant -> the MXU's / VPU's enables and TMEM's pw_*, 8-10
  // levels, +0.03 ns)
  logic                   qv, qlast;
  logic [LANES-1:0][31:0] qwl;
  logic [LANES-1:0]       qm;
  logic [LANES-1:0][31:0] qd;
  // RQ: the request register (stage a): the read's slice, lanes and per-lane addresses, and its
  // position (as, ar, acl) and write base (awa) for stage p
  logic                   av;
  logic [S-1:0]           asel;
  logic [LANES-1:0]       am;
  logic [LANES-1:0][31:0] aal;
  logic [31:0]            awa;

  // (qv is set only in C_RUN and cleared on leaving it: checked below)
  always_comb begin
    ack = (st == C_ACK);
    r_en = '0; r_addr = '0;
    w_en = '0; w_addr = '0; w_data = '0;
    for (int k = 0; k < S; k++)
      for (int l = 0; l < LANES; l++) begin
        if (RQ ? av && asel[k] && am[l] : rd_on && ssel[k] && rm[l]) r_en[k][l] = 1'b1;
        if (UG || r_en[k][l]) r_addr[k][l] = RQ ? aal[l] : ra + 32'(l);
      end
    for (int l = 0; l < LANES; l++) begin
      if (qv && qm[l]) w_en[l] = 1'b1;
      if (UG || w_en[l]) begin
        w_addr[l] = qwl[l];
        w_data[l] = qd[l];
      end
    end
  end

  // the issue stage's next read: its slice's stride (w4s), the next slice's first row (w1n)
  logic [31:0] w4s, w1n;
  always_comb begin
    w4s = '0;
    w1n = '0;
    for (int k = 0; k < S; k++) if (ssel[k]) w4s |= cmds[k].w4;
    for (int k = 1; k < S; k++) if (ssel[k - 1]) w1n |= cmds[k].w1;
  end

`ifndef SYNTHESIS
  // every slice names the same collective, and a GATHER the same dst (R[rb] + w2, resolved),
  // rows, cols, drs and seg (docs/isa.md): the unit takes slice 0's
  always @(posedge clk)
    if (!rst && st == C_IDLE && &req)
      for (int k = 1; k < S; k++)
        if (cmds[k].op != cmds[0].op ||
            (cmds[0].op == OP_GATHER && (cmds[k].w2 != cmds[0].w2 || cmds[k].w3 != cmds[0].w3 ||
                                         cmds[k].w5 != cmds[0].w5 || cmds[k].w6 != cmds[0].w6)))
          $fatal(1, "otpu_coll: slices 0 and %0d disagree on the collective", k);
`endif

  always_ff @(posedge clk) begin
    if (rst) begin
      st <= C_IDLE;
      pv <= 1'b0;
      qv <= 1'b0;
      rd_on <= 1'b0;
      av <= 1'b0;
    end else begin
      case (st)
        C_IDLE: if (&req) begin
          if (cmds[0].op == OP_BAR || cmds[0].w3[15:0] == 0 || cmds[0].w3[31:16] == 0) begin
            st <= C_ACK;
          end else begin
            dst  <= cmds[0].w2;
            rows <= cmds[0].w3[15:0];
            cols <= cmds[0].w3[31:16];
            drs  <= cmds[0].w5;
            seg  <= cmds[0].w6;
            s <= '0; r <= '0; c <= '0;
            ssel <= S'(1);
            rrow <= cmds[0].w1;
            ra <= cmds[0].w1;
            for (int l = 0; l < LANES; l++) begin
              rm[l] <= (l < int'(cmds[0].w3[31:16]));
              rm0[l] <= (l < int'(cmds[0].w3[31:16]));
            end
            c_l <= LANES >= int'(cmds[0].w3[31:16]);
            cl0 <= LANES >= int'(cmds[0].w3[31:16]);
            r_l <= cmds[0].w3[15:0] == 16'd1;
            rl0 <= cmds[0].w3[15:0] == 16'd1;
            wrow <= cmds[0].w2; wseg <= cmds[0].w2;
            pv <= 1'b0;
            qv <= 1'b0;
            av <= 1'b0;
            issuing <= 1'b1;
            rd_on <= 1'b1;
            st <= C_RUN;
          end
        end
        C_RUN: if (gnt) begin
          qv <= pv;
          for (int l = 0; l < LANES; l++) qwl[l] <= pwa + 32'(l);
          qm <= pm;
          for (int l = 0; l < LANES; l++) begin
            logic [31:0] d;
            d = '0;
            for (int k = 0; k < S; k++) if (psel[k]) d |= r_data[k][l];
            qd[l] <= d;
          end
          qlast <= pv && plast;
          if (RQ) begin                                 // the read granted is stage a's
            av <= issuing;
            asel <= ssel; am <= rm;
            for (int l = 0; l < LANES; l++) aal[l] <= ra + 32'(l);
            alast <= c_l && r_l && ssel[S-1];
            awa <= wrow + c;
            pv <= av;
            psel <= asel; plast <= alast;
            pwa <= awa;
            pm <= am;                                   // (rm[l] is c + l < cols)
          end else begin
            pv <= issuing;
            psel <= ssel; plast <= c_l && r_l && ssel[S-1];
            pwa <= wrow + c;
            pm <= rm;
          end
          // the next read (ra is rrow + c): the row's next chunk, the slice's next row, the next
          // slice's first row, or none
          if (issuing) begin
            if (c_l) begin
              c <= '0;
              c_l <= cl0;
              rm <= rm0;
              if (r_l) begin
                r <= '0;
                r_l <= rl0;
                if (ssel[S-1]) begin
                  issuing <= 1'b0;
                  rd_on <= 1'b0;
                  ra <= rrow;
                end else begin
                  s <= s + 1;
                  ssel <= ssel << 1;
                  rrow <= w1n;
                  ra <= w1n;
                  wseg <= wseg + seg;
                  wrow <= wseg + seg;
                end
              end else begin
                r <= r + 1;
                r_l <= r + 2 == 32'(rows);
                rrow <= rrow + w4s;
                ra <= rrow + w4s;
                wrow <= wrow + drs;
              end
            end else begin
              c <= c + LANES;
              c_l <= c + 2 * LANES >= 32'(cols);
              ra <= ra + LANES;
              for (int l = 0; l < LANES; l++) rm[l] <= (c + 32'(LANES + l) < 32'(cols));
            end
          end
          if (pv && plast)
            pv <= 1'b0;                                 // the last read's data moves to q
          if (qv && qlast) begin                         // the last write is taken this cycle
            qv <= 1'b0;
            st <= C_ACK;
          end
        end
        C_ACK:  st <= C_COOL;
        C_COOL: st <= C_IDLE;
        default: st <= C_IDLE;
      endcase
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && (rd_on != (st == C_RUN && issuing) || (rd_on && ssel != S'(1) << s)))
      $fatal(1, "otpu_coll: rd_on %0d / ssel %b do not match st %0d issuing %0d s %0d",
             rd_on, ssel, st, issuing, s);
  // qwl[l] is the registered pwa + l (qwa: the one address it replaced)
  logic [31:0] qwa;
  bit rst_seen;                                 // (registers start arbitrary)
  initial rst_seen = 1'b0;
  always @(posedge clk) begin
    if (rst) rst_seen <= 1'b1;
    else if (st == C_RUN && gnt) qwa <= pwa;
    if (!rst && rst_seen && st == C_RUN && qv)
      for (int l = 0; l < LANES; l++)
        if (qwl[l] != qwa + 32'(l))
          $fatal(1, "otpu_coll: lane %0d's write address %0d is not %0d", l, qwl[l], qwa + 32'(l));
  end
  // the gating's qv without st (qv only in C_RUN); the issue stage's flags and ra as they stand
  // for (c_l, r_l, rm: c + LANES >= cols, r + 1 == rows, c + l < cols; ra: rrow + c)
  always @(posedge clk)
    if (!rst && rst_seen) begin
      if (qv && st != C_RUN) $fatal(1, "otpu_coll: qv outside C_RUN (st %0d)", st);
      if (st == C_RUN && issuing)
        if (c_l != (c + LANES >= 32'(cols)) || r_l != (r + 1 == 32'(rows)) || ra != rrow + c)
          $fatal(1, "otpu_coll: c %0d r %0d (cols %0d rows %0d): c_l %0d r_l %0d, ra %0d rrow %0d",
                 c, r, cols, rows, c_l, r_l, ra, rrow);
      if (st == C_RUN && issuing)
        for (int l = 0; l < LANES; l++)
          if (rm[l] != (c + 32'(l) < 32'(cols)))
            $fatal(1, "otpu_coll: lane %0d's mask %0d at c %0d (cols %0d)", l, rm[l], c, cols);
    end
`endif
endmodule

// The collective unit without GATHER (otpu_board COLL_GATHER = 0, one slice only;
// docs/infra_hunt.md 5): BAR as otpu_coll runs it (acknowledged two cycles after the request,
// then a cycle cool-down), no TMEM access (its read and write ports tied off, so synthesis drops
// the TMEM's collective write port). A GATHER is not run: it is acknowledged like a BAR, moves no
// data, and raises gerr until reset (STATUS ERROR through otpu_board). The compiler never emits
// GATHER at S = 1 (opentpu.language.all_gather), and the host refuses to load a program with one
// on such a bitstream (CAPS2 bit0 = 0: opentpu/host/board.py). Ports as otpu_coll's, plus gerr.
module otpu_coll_bar
  import otpu_pkg::*;
#(
  parameter int S     = 1,
  parameter int LANES = 8
) (
  input  logic                            clk,
  input  logic                            rst,
  input  logic [S-1:0]                    req,
  input  cmd_t                            cmds [S],
  input  logic                            gnt,
  output logic                            ack,
  output logic [S-1:0][LANES-1:0]         r_en,
  output logic [S-1:0][LANES-1:0][31:0]   r_addr,
  input  logic [S-1:0][LANES-1:0][31:0]   r_data,
  output logic [LANES-1:0]                w_en,
  output logic [LANES-1:0][31:0]          w_addr,
  output logic [LANES-1:0][31:0]          w_data,
  output logic                            gerr      // a GATHER came (sticky until rst)
);
  initial if (S != 1) $fatal(1, "otpu_coll_bar: one slice only (GATHER across slices needs otpu_coll)");
  typedef enum logic [1:0] {C_IDLE, C_ACK, C_COOL} st_t;
  st_t st;
  assign ack    = (st == C_ACK);
  assign r_en   = '0;
  assign r_addr = '0;
  assign w_en   = '0;
  assign w_addr = '0;
  assign w_data = '0;
  always_ff @(posedge clk) begin
    if (rst) begin
      st <= C_IDLE;
      gerr <= 1'b0;
    end else begin
      case (st)
        C_IDLE: if (&req) begin
          st <= C_ACK;
          if (cmds[0].op == OP_GATHER) gerr <= 1'b1;
        end
        C_ACK:  st <= C_COOL;
        default: st <= C_IDLE;
      endcase
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && st == C_IDLE && &req && cmds[0].op == OP_GATHER && !$test$plusargs("coll_gerr_ok"))
      $fatal(1, "otpu_coll_bar: GATHER on a collective unit without it (COLL_GATHER = 0)");
`endif
endmodule

// The collective unit's side of the SLR crossings (otpu_board at S > 1, every slice a few
// register stages away). otpu_coll runs unchanged against it, but its `gnt` is this hub's, from
// the hub's own registers: no slice's TMEM grant in the cycle (at S = 1 the grant is the AND of
// the slices' grants in the same cycle, a path through every SLR). Each slice's otpu_coll_port
// takes the unit's reads and writes and has them granted in its own TMEM, later. A read holds the
// unit until its data is back (the unit takes a read's data at its next granted cycle: one read
// in flight); writes go to every slice's port, at most F not yet written per port (each port's
// pops come back as credits). The unit's ack goes to the slices once every port has written
// everything, and a slice's request counts again only after it dropped (the unit is idle two
// cycles after its ack; the slices see the ack later). The ISA's rule that a GATHER's source and
// destination do not overlap makes the order of a slice's reads and writes free.
module otpu_coll_hub
  import otpu_pkg::*;
#(
  parameter int S     = 2,
  parameter int LANES = 8,
  parameter int F     = 8               // writes a port holds
) (
  input  logic                            clk,
  input  logic                            rst,
  // otpu_coll's ports
  output logic [S-1:0]                    u_req,
  output cmd_t                            u_cmds [S],
  output logic                            u_gnt,
  input  logic                            u_ack,
  input  logic [S-1:0][LANES-1:0]         u_ren,
  input  logic [S-1:0][LANES-1:0][31:0]   u_raddr,
  output logic [S-1:0][LANES-1:0][31:0]   u_rdata,
  input  logic [LANES-1:0]                u_wen,
  input  logic [LANES-1:0][31:0]          u_waddr,
  input  logic [LANES-1:0][31:0]          u_wdata,
  // the slices' ports, through the crossing registers
  input  logic [S-1:0]                    x_req,
  input  cmd_t                            x_cmds [S],
  output logic                            x_ack,     // to every slice (pulse)
  output logic [S-1:0]                    x_rv,      // a read for slice s (pulse)
  output logic [LANES-1:0]                x_rm,
  output logic [LANES-1:0][31:0]          x_ra,
  input  logic [S-1:0]                    x_dv,      // its data (pulse)
  input  logic [S-1:0][LANES-1:0][31:0]   x_dd,
  output logic                            x_wv,      // a write for every slice (pulse)
  output logic [LANES-1:0]                x_wm,
  output logic [LANES-1:0][31:0]          x_wa,
  output logic [LANES-1:0][31:0]          x_wd,
  input  logic [S-1:0]                    x_pop      // a port wrote one (pulse)
);
  localparam int CW = $clog2(F + 2);
  logic                   rd_wait, pend;
  logic [S-1:0]           blk;            // the slice's request was acknowledged, not yet dropped
  logic [S-1:0][CW-1:0]   o;              // writes sent to a port and not popped (as seen here)
  logic [LANES-1:0][31:0] hd;
  logic                   room, drained;

  always_comb begin
    room = 1'b1;
    drained = 1'b1;
    for (int k = 0; k < S; k++) begin
      if (o[k] >= CW'(F)) room = 1'b0;
      if (o[k] != '0) drained = 1'b0;
    end
  end
  assign u_gnt = !rd_wait && room;
  assign u_req = rst ? '0 : x_req & ~blk;   // (a request still crossing when the run stops)
  for (genvar k = 0; k < S; k++) begin : g_k
    assign u_cmds[k] = x_cmds[k];
    assign u_rdata[k] = hd;
  end
  always_comb begin
    x_rv = '0;
    x_rm = '0;
    x_ra = '0;
    for (int k = 0; k < S; k++) begin   // the unit reads one slice at a time (the others' '0)
      x_rv[k] = u_gnt && |u_ren[k];
      x_rm |= u_ren[k];
      x_ra |= u_raddr[k];
    end
    x_wv = u_gnt && |u_wen;
    x_wm = u_wen;
    x_wa = u_waddr;
    x_wd = u_wdata;
    x_ack = pend && drained;
  end

  always_ff @(posedge clk) begin
    for (int k = 0; k < S; k++) if (x_dv[k]) hd <= x_dd[k];
    if (rst) begin
      rd_wait <= 1'b0;
      pend <= 1'b0;
      blk <= '0;
      o <= '0;
    end else begin
      if (|x_rv) rd_wait <= 1'b1;
      else if (|x_dv) rd_wait <= 1'b0;
      if (u_ack) pend <= 1'b1;
      else if (x_ack) pend <= 1'b0;
      for (int k = 0; k < S; k++) begin
        if (u_ack) blk[k] <= 1'b1;
        else if (!x_req[k]) blk[k] <= 1'b0;
        o[k] <= o[k] + CW'(x_wv) - CW'(x_pop[k]);
      end
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && ((|x_rv && rd_wait) || (|x_dv && !rd_wait) || $countones(x_rv) > 1))
      $fatal(1, "otpu_coll_hub: read %b while waiting %0d, data %b", x_rv, rd_wait, x_dv);
`endif
endmodule

// A slice's collective port at S > 1 (otpu_board): the hub's reads (one at a time) and writes (a
// FIFO of F) presented on the slice's collective TMEM port and granted by the slice's own TMEM
// arbiter (gnt = gnt_local: the collective group's grant, reads and writes together), the read
// data returned the cycle after it is fresh, each written entry returned as a pop.
// Everything the slice's arbiter sees comes from flip-flops: the held read's lanes (hr, the
// read's enables, registered with hv / hq) and address, and the write FIFO's head (h_*: its lanes,
// zero while there is none, its addresses and data), loaded from the LUT RAM or, the FIFO empty,
// straight from the hub's write (250 MHz, FK35: u_cport/rp -> the LUT RAM's read -> the slice's
// TMEM bank masks and priority -> its MXU grant, fanout 2,096). The grant clears the head's valid
// and lanes only; its addresses and data load from flip-flop state (the head empty), so a taken
// entry's successor shows a cycle later than from the RAM before (a write every other cycle at
// most, against one read in flight per GATHER beat: no GATHER is slower for it). The FIFO with
// its head holds F entries, as the hub's credits count.
module otpu_coll_port #(
  parameter int LANES = 8,
  parameter int F     = 8
) (
  input  logic                    clk,
  input  logic                    rst,
  // from the hub, through the crossing registers
  input  logic                    rv,
  input  logic [LANES-1:0]        rm,
  input  logic [LANES-1:0][31:0]  ra,
  input  logic                    wv,
  input  logic [LANES-1:0]        wm,
  input  logic [LANES-1:0][31:0]  wa,
  input  logic [LANES-1:0][31:0]  wd,
  // to the hub
  output logic                    dv,
  output logic [LANES-1:0][31:0]  dd,
  output logic                    pop,
  // the slice's collective port (otpu_slice coll_*)
  output logic [LANES-1:0]        coll_ren,
  output logic [LANES-1:0][31:0]  coll_raddr,
  input  logic [LANES-1:0][31:0]  coll_rdata,
  output logic [LANES-1:0]        coll_wen,
  output logic [LANES-1:0][31:0]  coll_waddr,
  output logic [LANES-1:0][31:0]  coll_wdata,
  input  logic                    gnt_local,
  output logic                    gnt
);
  localparam int PW = $clog2(F);         // F a power of two: the pointers wrap
  // hq: hv's second cycle. The held read is asked from it (TMEM's input stage, release r2,
  // docs/tmem_nocopy.md 7: every unit's read goes to TMEM a cycle after it forms, the writes a
  // cycle after their grant), and held there until granted, with no replay
  logic                   hv, hq, got;
  logic [LANES-1:0]       hm, hr;           // hr: coll_ren, (hv && hq) ? hm : 0
  logic [LANES-1:0][31:0] ha;
  logic [LANES-1:0]       fm [F];
  logic [LANES-1:0][31:0] fa [F];
  logic [LANES-1:0][31:0] fd [F];
  logic [PW-1:0]          wp, rp;
  logic [PW:0]            n;                // entries in the RAM (the head's not counted)
  logic                   h_v;              // the head
  logic [LANES-1:0]       h_wen;
  logic [LANES-1:0][31:0] h_wa, h_wd;
  wire                    wtake = h_v && gnt_local;
  wire                    hload = !h_v && (n != '0 || wv);   // the head takes the next entry
  wire                    hbyp = n == '0;                    // ... the hub's write itself

  assign gnt = gnt_local;
  assign coll_ren = hr;
  assign coll_raddr = ha;
  assign coll_wen = h_wen;
  assign coll_waddr = h_wa;
  assign coll_wdata = h_wd;

  always_ff @(posedge clk) begin
    if (got) dd <= coll_rdata;          // fresh the cycle after the granted read
    if (rv) begin
      hm <= rm;
      ha <= ra;
    end
    if (wv && !(hload && hbyp)) begin
      fm[wp] <= wm;
      fa[wp] <= wa;
      fd[wp] <= wd;
    end
    if (hload) begin
      h_wa <= hbyp ? wa : fa[rp];
      h_wd <= hbyp ? wd : fd[rp];
    end
    if (rst) begin
      hv <= 1'b0;
      hq <= 1'b0;
      hr <= '0;
      got <= 1'b0;
      dv <= 1'b0;
      pop <= 1'b0;
      wp <= '0;
      rp <= '0;
      n <= '0;
      h_v <= 1'b0;
      h_wen <= '0;
    end else begin
      got <= hv && hq && gnt_local;
      dv <= got;
      pop <= wtake;
      if (rv) hv <= 1'b1;
      else if (hv && hq && gnt_local) hv <= 1'b0;
      hq <= hv && !(hq && gnt_local);
      // the read's lanes next cycle: hv and hq both then (rv only with hv low: hq low then)
      hr <= (hv && !(hq && gnt_local)) ? hm : '0;
      if (wv && !(hload && hbyp)) wp <= wp + 1'b1;
      if (hload && !hbyp) rp <= rp + 1'b1;
      n <= n + (PW+1)'(wv && !(hload && hbyp)) - (PW+1)'(hload && !hbyp);
      if (wtake) begin
        h_v <= 1'b0;
        h_wen <= '0;
      end else if (hload) begin
        h_v <= 1'b1;
        h_wen <= hbyp ? wm : fm[rp];
      end
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && ((rv && hv) || (wv && !(hload && hbyp) && n == (PW+1)'(F) && !hload) ||
                 n + (PW+1)'(h_v) > (PW+1)'(F) || hr != ((hv && hq) ? hm : '0) ||
                 (!h_v && h_wen != '0)))
      $fatal(1, "otpu_coll_port: a read while one is held (%0d), a write into a full FIFO (%0d + %0d) or a stale head",
             hv, n, h_v);
`endif
endmodule
