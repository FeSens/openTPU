// Collective unit: BAR and GATHER across S slices. When every slice is waiting on the same
// collective, GATHER streams each slice's segment, up to LANES consecutive words per cycle, and
// broadcasts it into the TMEM of all slices at dst + s*seg + r*drs + c; then all slices are
// released together. `gnt` is the AND of every slice's TMEM grant; without it the unit holds.
// RQ (TMEM's input stage, release r2, docs/tmem_nocopy.md 7): the reads go to the slices from a
// request register (stage a), a cycle after they form, as every unit's do, and a read's data is
// taken one granted cycle later than without it. For the slices' TMEMs on the unit's own ports
// (otpu_top; otpu_board at S = 1); otpu_board's hub at S > 1 takes RQ 0 (the ports present).
module otpu_coll
  import otpu_pkg::*;
#(
  parameter int S     = 1,
  parameter int LANES = 8,
  parameter bit RQ    = 1'b1
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
  logic [31:0] s, r, c, ps, pr, pcl;
  logic        pv, issuing;
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
  logic [31:0]            as_, ar, acl, awa;

  always_comb begin
    ack = (st == C_ACK);
    r_en = '0; r_addr = '0;
    w_en = '0; w_addr = '0; w_data = '0;
    for (int k = 0; k < S; k++)
      for (int l = 0; l < LANES; l++)
        if (RQ ? av && asel[k] && am[l] : rd_on && ssel[k] && rm[l]) begin
          r_en[k][l] = 1'b1;
          r_addr[k][l] = RQ ? aal[l] : ra + 32'(l);
        end
    if (st == C_RUN && qv) begin
      for (int l = 0; l < LANES; l++)
        if (qm[l]) begin
          w_en[l] = 1'b1;
          w_addr[l] = qwl[l];
          w_data[l] = qd[l];
        end
    end
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
            for (int l = 0; l < LANES; l++) rm[l] <= (l < int'(cmds[0].w3[31:16]));
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
          for (int l = 0; l < LANES; l++)
            for (int k = 0; k < S; k++) if (k == int'(ps)) qd[l] <= r_data[k][l];
          qlast <= pv && ps + 1 == S && pr + 1 == 32'(rows) && pcl + LANES >= 32'(cols);
          if (RQ) begin                                 // the read granted is stage a's
            av <= issuing;
            asel <= ssel; am <= rm;
            for (int l = 0; l < LANES; l++) aal[l] <= ra + 32'(l);
            as_ <= s; ar <= r; acl <= c;
            awa <= wrow + c;
            pv <= av;
            ps <= as_; pr <= ar; pcl <= acl;
            pwa <= awa;
            pm <= am;                                   // (rm[l] is c + l < cols)
          end else begin
            pv <= issuing;
            ps <= s; pr <= r; pcl <= c;
            pwa <= wrow + c;
            for (int l = 0; l < LANES; l++) pm[l] <= (c + 32'(l) < 32'(cols));
          end
          if (issuing) begin
            logic [31:0] nc, nrrow;
            nc = c; nrrow = rrow;
            if (c + LANES >= 32'(cols)) begin
              nc = '0;
              if (r + 1 == 32'(rows)) begin
                r <= '0;
                if (s + 1 == S) begin
                  issuing <= 1'b0;
                  rd_on <= 1'b0;
                end else begin
                  s <= s + 1;
                  ssel <= ssel << 1;
                  for (int k = 0; k < S; k++) if (k == int'(s) + 1) nrrow = cmds[k].w1;
                  wseg <= wseg + seg;
                  wrow <= wseg + seg;
                end
              end else begin
                r <= r + 1;
                for (int k = 0; k < S; k++) if (k == int'(s)) nrrow = rrow + cmds[k].w4;
                wrow <= wrow + drs;
              end
            end else begin
              nc = c + LANES;
            end
            c <= nc; rrow <= nrrow;
            ra <= nrrow + nc;
            for (int l = 0; l < LANES; l++) rm[l] <= (nc + 32'(l) < 32'(cols));
          end
          if (pv && ps + 1 == S && pr + 1 == 32'(rows) && pcl + LANES >= 32'(cols))
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
// takes the unit's reads and writes and has them granted in its own TMEM, later. Writes go to
// every slice's port, at most F not yet written per port (each port's pops come back as
// credits). The unit's ack goes to the slices once every port has written everything, and a
// slice's request counts again only after it dropped (the unit is idle two cycles after its ack;
// the slices see the ack later). The ISA's rule that a GATHER's source and destination do not
// overlap makes the order of a slice's reads and writes free (checked in simulation, below).
// NR 1 (otpu_board COLL_PIPE 0): a read holds the unit until its data is back (the unit takes a
// read's data at its next granted cycle: one read in flight, a round trip through the crossings
// each). NR > 1 (COLL_PIPE 1): up to NR reads in flight, the unit granted on their credits. The
// unit then issues a read every granted cycle; its control never looks at the read data, so its
// writes (one a read, in its reads' order) are taken as each read's lanes and addresses (u_wdata
// is not used) and the hub sends each read's write itself, with the read's data, once that is
// back: the reads' slices in issue order (om), the unit's writes (wmm, wam), each slice's data in
// its reads' order (g_rf: two slices' data can come in the same cycle), and the oldest read's
// write from flip-flops (wo_*) when its data is in and every port has room. A read's credit
// comes back when its write goes. So each slice sees the reads and the writes it sees at NR 1,
// in the same order, with the same data; only when differs. NR covers the read's round trip
// (otpu_board: 2 XRS + 6 cycles) and F the write's credit loop (2 XRS + 4), so both run at a read
// and a write a cycle.
// NC > 1 (otpu_board NC: the cards' links; NR > 1, LANES 8): this card's slices are global
// slices S cid .. S cid + S - 1 of S ncards on ncards cards (otpu_ctrl's CARD; ncards <= NC),
// its hub joined to the other cards' by two link ports (otpu_coll_lx each, qsfp-link's lt_port
// behind it). The unit runs this card's slices as at one card, with dst + S cid seg for dst (the
// hub's w2, two cycles after the request: its product, then its sum, from registers); otpu_coll_xc
// sends each local write's words to the other cards as it goes to the write stage, and gives the
// write stage the other cards' words in the cycles the oldest read's write does not take it. The
// slices' ack waits for both. ncards 1 runs the card alone, as at NC 1 (two cycles later a
// collective). NC 1 builds the hub as before: the link ports unused, its outputs 0.
module otpu_coll_hub
  import otpu_pkg::*;
#(
  parameter int S     = 2,
  parameter int LANES = 8,
  parameter int F     = 8,              // writes a port holds
  parameter int NR    = 1,              // reads in flight (1, or a power of two >= 4)
  parameter int NC    = 1,              // cards at most (1: no links)
  parameter int PKT   = 32,             // NC > 1: a message's payload words at most
  parameter int TXC   = 16,             // NC > 1: a link adapter's TX words (the hub's credits)
  parameter int RXC   = 16              // NC > 1: the hub's RX words a link (the adapter's credits)
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
  input  logic [S-1:0]                    x_pop,     // a port wrote one (pulse)
  output logic                            x_wt,      // FOLLOW: the write advances the top (NC 1: 1)
  // the cards (NC > 1): the links' reset (the board's or X2_CTRL LRST), X2_CARD, the watchdog's
  // cycles (COLL_WD, 0: off), the links' crossings (otpu_coll_lx: port 0 left, 1 right), COLL_STATUS
  // and the error it raises, X2_SEQ
  input  logic                            lrst,
  input  logic [7:0]                      cid,
  input  logic [7:0]                      ncards,
  input  logic [31:0]                     wdog,
  output logic [1:0]                      l_tv,      // a TX word (pulse)
  output logic [1:0]                      l_tlast,
  output logic [1:0][255:0]               l_td,
  output logic [1:0]                      l_rcred,   // an RX word taken (pulse)
  output logic                            l_abort,   // send ABORT (pulse, both links)
  input  logic [1:0]                      l_tcred,   // a TX word went to the port (pulse)
  input  logic [1:0]                      l_rv,      // an RX word (pulse)
  input  logic [1:0]                      l_rhdr,
  input  logic [1:0]                      l_rlast,
  input  logic [1:0][255:0]               l_rd,
  input  logic [1:0]                      l_up,
  input  logic [1:0]                      l_rerr,    // a message failed its check (pulse)
  input  logic [1:0]                      l_abin,    // the peer sent ABORT (pulse)
  output logic [7:0]                      xst,
  output logic                            xerr,
  output logic [15:0]                     xseq
);
  initial if (NR != 1 && (NR < 4 || (NR & (NR - 1)) != 0))
    $fatal(1, "otpu_coll_hub: NR %0d (1, or a power of two >= 4: a read's write is the unit's two granted cycles later)", NR);
  initial if (NC < 1 || NC > 8 || (NC > 1 && (NR == 1 || LANES != 8 || S < 2)))
    $fatal(1, "otpu_coll_hub: NC %0d (1 .. 8; > 1: NR > 1, LANES 8, S >= 2)", NC);
  localparam int CW = $clog2(F + 2);
  logic                   pend;
  logic [S-1:0]           blk;            // the slice's request was acknowledged, not yet dropped
  logic [S-1:0][CW-1:0]   o;              // writes sent to a port and not popped (as seen here)
  logic                   room, drained;
  logic                   a_rd, a_wr;     // a read sent, a write taken from the unit (the checks)
  // the write that goes to the write stage this cycle (the checks; NC > 1 the other cards' too)
  logic                   s_wv, s_rw;     //   a write; one from another card
  logic [LANES-1:0]       s_wm;
  logic [LANES-1:0][31:0] s_wa;

  always_comb begin
    room = 1'b1;
    drained = 1'b1;
    for (int k = 0; k < S; k++) begin
      if (o[k] >= CW'(F)) room = 1'b0;
      if (o[k] != '0) drained = 1'b0;
    end
  end
  if (NC == 1) begin : g_c1
    assign u_req = rst ? '0 : x_req & ~blk;   // (a request still crossing when the run stops)
    for (genvar k = 0; k < S; k++) begin : g_k
      assign u_cmds[k] = x_cmds[k];
    end
  end else begin : g_cn
    // dst + S cid seg for this card's segments (seg x S cid, then the sum, from registers: the
    // command holds while its request does); the request reaches the unit two cycles late
    logic [S-1:0]       rq1, rq2;
    logic [S-1:0][31:0] m1, w2r;
    always_ff @(posedge clk) begin
      for (int k = 0; k < S; k++) begin
        m1[k] <= x_cmds[k].w6 * 32'(S * int'(cid));
        w2r[k] <= x_cmds[k].w2 + m1[k];
      end
      rq1 <= rst ? '0 : x_req;
      rq2 <= rst ? '0 : rq1 & x_req;
    end
    assign u_req = rst ? '0 : rq2 & x_req & ~blk;
    for (genvar k = 0; k < S; k++) begin : g_k
      always_comb begin
        u_cmds[k] = x_cmds[k];
        u_cmds[k].w2 = w2r[k];
      end
    end
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
  end
  assign a_rd = |x_rv;

  if (NR == 1) begin : g_one
    logic                   rd_wait;
    logic [LANES-1:0][31:0] hd;
    assign u_gnt = !rd_wait && room;
    for (genvar k = 0; k < S; k++) begin : g_k
      assign u_rdata[k] = hd;
    end
    always_comb begin
      x_wv = u_gnt && |u_wen;
      x_wm = u_wen;
      x_wa = u_waddr;
      x_wd = u_wdata;
      x_ack = pend && drained;
    end
    assign a_wr = x_wv;
    assign s_wv = x_wv;
    assign s_rw = 1'b0;
    assign s_wm = u_wen;
    assign s_wa = u_waddr;
    assign {l_tv, l_tlast, l_td, l_rcred, l_abort, xst, xerr, xseq} = '0;
    assign x_wt = 1'b1;

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
  end else begin : g_pipe
    localparam int NW = $clog2(NR);
    localparam int SW = (S > 1) ? $clog2(S) : 1;
    logic [NW:0]            nout;         // reads out: sent, their write not yet (the credits)
    logic                   g_ok;         // nout < NR, from a register
    logic [SW-1:0]          om [NR];      // each read's slice, in issue order
    logic [NW-1:0]          ow, orp;
    logic [NW:0]            on;
    logic [LANES-1:0]       wmm [NR];     // the unit's write for each read: its lanes,
    logic [LANES-1:0][31:0] wam [NR];     //   their addresses
    logic [NW-1:0]          ww, wrp;
    logic [NW:0]            wn;
    logic [S-1:0][NW-1:0]   rw, rrp;      // each slice's data FIFO (g_rf)
    logic [S-1:0][NW:0]     rn;
    logic [S-1:0][LANES-1:0][31:0] rhd;   //   its head
    logic                   wo_v;         // the write stage: the oldest read's write
    logic [LANES-1:0]       wo_m;
    logic [LANES-1:0][31:0] wo_a, wo_d;
    logic [SW-1:0]          rs, hs;       // the read's slice; the oldest read's
    logic                   dr;           // the oldest read's write goes to the write stage
    // NC > 1 (otpu_coll_xc): the links' TX has room for slice k's words (c_tok); another card's
    // write (c_wv: its lanes, addresses, words), taken by the write stage (c_wt); every remote
    // piece of the collective in (c_rdone); NC 1: none (constants)
    logic [S-1:0]           c_tok;
    logic                   c_wv, c_wt, c_rdone, c_act;
    logic [LANES-1:0]       c_wm;
    logic [LANES-1:0][31:0] c_wa, c_wd;

    assign u_gnt = g_ok;
    for (genvar k = 0; k < S; k++) begin : g_k
      assign u_rdata[k] = '0;             // (the unit's write data: not used)
    end
    assign a_wr = u_gnt && |u_wen;
    always_comb begin
      rs = '0;
      for (int k = 0; k < S; k++) if (|u_ren[k]) rs = SW'(k);
    end
    assign hs = om[orp];
    assign dr = on != '0 && wn != '0 && rn[hs] != '0 && room && c_tok[hs];
    assign c_wt = c_wv && room && !dr;
    always_comb begin
      x_wv = wo_v;
      x_wm = wo_m;
      x_wa = wo_a;
      x_wd = wo_d;
      x_ack = pend && nout == '0 && drained && c_rdone;
    end
    assign s_wv = dr || c_wt;
    assign s_rw = c_wt;
    assign s_wm = dr ? wmm[wrp] : c_wm;
    assign s_wa = dr ? wam[wrp] : c_wa;

    if (NC == 1) begin : g_c1
      assign c_tok = '1;
      assign {c_wv, c_wm, c_wa, c_wd, c_act} = '0;
      assign c_rdone = 1'b1;
      assign {l_tv, l_tlast, l_td, l_rcred, l_abort, xst, xerr, xseq} = '0;
      assign x_wt = 1'b1;
    end else begin : g_cn
      logic wo_t;             // FOLLOW: the write stage's write advances the top
      otpu_coll_xc #(.S(S), .NC(NC), .PKT(PKT), .TXC(TXC), .RXC(RXC)) u_xc (
        .clk, .rst, .lrst, .cid, .ncards, .wdog,
        .go(&u_req && !c_act), .cmd(x_cmds[0]), .done(x_ack), .act(c_act), .rdone(c_rdone),
        .tv(dr), .tk(hs), .td(rhd[hs]), .tok(c_tok),
        .wv(c_wv), .wm(c_wm), .wa(c_wa), .wd(c_wd), .wt(c_wt),
        .l_tv, .l_tlast, .l_td, .l_rcred, .l_abort, .l_tcred, .l_rv, .l_rhdr, .l_rlast, .l_rd,
        .l_up, .l_rerr, .l_abin, .xst, .xerr, .xseq);
      always_ff @(posedge clk)
        if (dr || c_wt) wo_t <= dr;
      assign x_wt = wo_t;
    end

    for (genvar k = 0; k < S; k++) begin : g_rf
      logic [LANES-1:0][31:0] m [NR];
      always_ff @(posedge clk) if (x_dv[k]) m[rw[k]] <= x_dd[k];
      assign rhd[k] = m[rrp[k]];
    end
    always_ff @(posedge clk) if (a_rd) om[ow] <= rs;
    always_ff @(posedge clk) if (a_wr) wmm[ww] <= u_wen;
    always_ff @(posedge clk) if (a_wr) wam[ww] <= u_waddr;
    always_ff @(posedge clk) begin
      if (dr) begin
        wo_m <= wmm[wrp];
        wo_a <= wam[wrp];
        wo_d <= rhd[hs];
      end else if (c_wt) begin
        wo_m <= c_wm;
        wo_a <= c_wa;
        wo_d <= c_wd;
      end
      if (rst) begin
        pend <= 1'b0;
        blk <= '0;
        o <= '0;
        nout <= '0;
        g_ok <= 1'b1;
        ow <= '0; orp <= '0; on <= '0;
        ww <= '0; wrp <= '0; wn <= '0;
        rw <= '0; rrp <= '0; rn <= '0;
        wo_v <= 1'b0;
      end else begin
        logic [NW:0] nn;
        if (u_ack) pend <= 1'b1;
        else if (x_ack) pend <= 1'b0;
        nn = nout + (NW+1)'(a_rd) - (NW+1)'(dr);
        nout <= nn;
        g_ok <= nn < (NW+1)'(NR);
        if (a_rd) ow <= ow + 1'b1;
        if (dr) orp <= orp + 1'b1;
        on <= on + (NW+1)'(a_rd) - (NW+1)'(dr);
        if (a_wr) ww <= ww + 1'b1;
        if (dr) wrp <= wrp + 1'b1;
        wn <= wn + (NW+1)'(a_wr) - (NW+1)'(dr);
        for (int k = 0; k < S; k++) begin
          logic pk;
          pk = dr && hs == SW'(k);
          if (x_dv[k]) rw[k] <= rw[k] + 1'b1;
          if (pk) rrp[k] <= rrp[k] + 1'b1;
          rn[k] <= rn[k] + (NW+1)'(x_dv[k]) - (NW+1)'(pk);
          if (u_ack) blk[k] <= 1'b1;
          else if (!x_req[k]) blk[k] <= 1'b0;
          o[k] <= o[k] + CW'(dr || c_wt) - CW'(x_pop[k]);
        end
        wo_v <= dr || c_wt;
      end
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst) begin
        if ($countones(x_rv) > 1 || (a_rd && on == (NW+1)'(NR)) || (a_wr && wn == (NW+1)'(NR)))
          $fatal(1, "otpu_coll_hub: reads %b with %0d out, a write with %0d held", x_rv, on, wn);
        for (int k = 0; k < S; k++)
          if (x_dv[k] && rn[k] == (NW+1)'(NR))
            $fatal(1, "otpu_coll_hub: slice %0d's data with its FIFO full", k);
      end
`endif
  end

`ifndef SYNTHESIS
  // The ISA keeps a GATHER's source and destination apart (docs/isa.md), so a slice's reads may
  // pass the writes before them (NR > 1 lets them; NR 1 hardly). Checked: a read for slice k
  // touches no word a write still on its way to slice k's TMEM writes (taken from the unit, not
  // yet popped by k's port), and a write no word a read still out reads (sent, its data not back).
  // NC > 1: the writes in flight are taken as they go to the write stage (the other cards' among
  // the unit's, in the ports' order), and another card's write is checked against the reads
  // there. +coll_overlap_ok: not checked (a test of the check alone)
  typedef struct packed { logic [LANES-1:0] m; logic [LANES-1:0][31:0] a; } acc_t;
  acc_t ck_w [S][$];                      // per port: writes taken, not yet popped
  acc_t ck_r [S][$];                      // per slice: reads sent, their data not back
  bit   ck_off;
  initial ck_off = $test$plusargs("coll_overlap_ok");
  function automatic bit ck_hit(acc_t x, acc_t y);
    for (int i = 0; i < LANES; i++)
      for (int j = 0; j < LANES; j++)
        if (x.m[i] && y.m[j] && x.a[i] == y.a[j]) return 1'b1;
    return 1'b0;
  endfunction
  always @(posedge clk)
    if (rst) begin
      for (int k = 0; k < S; k++) begin ck_w[k].delete(); ck_r[k].delete(); end
    end else begin
      acc_t rd, wr, sw;
      rd.m = x_rm; rd.a = x_ra;
      wr.m = u_wen; wr.a = u_waddr;
      sw.m = s_wm; sw.a = s_wa;
      if (!ck_off) begin
        for (int k = 0; k < S; k++) begin
          if (x_rv[k])
            foreach (ck_w[k][i])
              if (ck_hit(rd, ck_w[k][i]))
                $fatal(1, "otpu_coll_hub: GATHER source / destination overlap: slice %0d reads a word a write still in flight writes (%0d ...)", k, rd.a[0]);
          if (a_wr) begin
            foreach (ck_r[k][i])
              if (ck_hit(wr, ck_r[k][i]))
                $fatal(1, "otpu_coll_hub: GATHER source / destination overlap: a write over a word slice %0d's read still in flight reads (%0d ...)", k, wr.a[0]);
            if (x_rv[k] && ck_hit(wr, rd))
              $fatal(1, "otpu_coll_hub: GATHER source / destination overlap: slice %0d's read and a write, the same cycle (%0d ...)", k, rd.a[0]);
          end
          if (s_rw) begin
            foreach (ck_r[k][i])
              if (ck_hit(sw, ck_r[k][i]))
                $fatal(1, "otpu_coll_hub: GATHER source / destination overlap: another card's write over a word slice %0d's read still in flight reads (%0d ...)", k, sw.a[0]);
            if (x_rv[k] && ck_hit(sw, rd))
              $fatal(1, "otpu_coll_hub: GATHER source / destination overlap: slice %0d's read and another card's write, the same cycle (%0d ...)", k, rd.a[0]);
          end
        end
      end
      for (int k = 0; k < S; k++) begin
        if (x_dv[k] && ck_r[k].size() != 0) void'(ck_r[k].pop_front());
        if (x_pop[k] && ck_w[k].size() != 0) void'(ck_w[k].pop_front());
        if (x_rv[k]) ck_r[k].push_back(rd);
        if (NC == 1 ? a_wr : s_wv) ck_w[k].push_back(NC == 1 ? wr : sw);
      end
    end
`endif
endmodule

// The hub's side of the cards' links (otpu_coll_hub at NC > 1). Collective n (go .. done: the
// card's slices issued it, the hub acknowledged it) runs on every card at once; its pieces are
// the global slices', g = S cid + k for this card's slice k.
// TX: each local write's words, as they go to the write stage (tv: slice tk, in the unit's order:
// slice, row, LANES words), go into the TX queue of each port its piece takes, with the message
// they start or continue: a message is a header word and up to PKT words of one piece (TYPE
// LK_DATA, LEN, ARG the collective's number, AUX LK_PIECE; the piece g, this card, the hops, the
// row and column of its first word: otpu_pkg LKH_*). The other ncards - 1 cards split between
// the two directions, the larger half left for an even k, right for an odd one: at two cards
// slice 0's piece goes out port 0 and slice 1's out port 1, a hop each (the ports forward a
// message on with HOPS > 1: the ring, later). A BAR (or a GATHER of no words) sends one token, a
// header of LEN 0, as slice 0's piece would go. Each queue sends its head on the port's credits
// (TXC: the adapter's words), its header first. A slice's write waits for room in its ports'
// queues (tok).
// RX: each port's words into a FIFO of RXC (a credit back to the adapter as each is taken),
// parsed in order: a header of collective n is taken only while n runs here (act), so another
// card's words for n + 1, sent once it issued n + 1, wait in the FIFO and in the port's RX FIFO
// behind it (the link's credits) until this card's slices issue n + 1: no remote write lands
// over a buffer this card still reads (the hazard the one card's common release rules out).
// A piece's first header starts the walk of its words (dst + g seg, then rows by drs and LANES
// words by LANES: the unit's own order), the next ones must continue it; each word is a write
// candidate for the write stage (ports in turn), whose lanes and addresses the walk gives.
// rdone once every other card's pieces (GATHER: (ncards - 1) S) or tokens (BAR: ncards - 1) are
// in, registered a cycle after the last one's write went to the write stage (so the ports' count
// has it). Errors: a header out of place, of another collective than n or n + 1, a piece seen
// twice or not another card's, a link's check (rx_err: a CRC, the RX FIFO's overflow, the link
// dropped), or the watchdog (wdog cycles of a collective waiting on the links with no word in
// and none out, 0 off), abort the collectives (xab): ABORT goes out both links once, every word
// received is dropped, the message on each link is ended (its words, or zeros) and no more are
// sent, and each collective completes on its local writes alone, so the program runs to its
// HALT; ABORT from a link does the same (and goes on out both links once). COLL_STATUS (xst): bit0 aborted, bit1 the watchdog, bit2 a
// header out of place, bit3 a link's check, bit4 ABORT from a link, bits 6:5 the links up; xerr
// (STATUS ERROR) on bit0 or bit3. FOLLOW in phase 1 tags only local writes; a follower
// must read only its card's local half. The remote pieces may arrive in any order.
// Resets: the links' side (both FIFOs, the TX queues and the message on each link, the parsers,
// the collectives' numbers, the abort and COLL_STATUS) resets with lrst alone, never with the run's
// rst (RUN 0), so a card whose run starts later finds the other card's words waiting, and the
// numbers (X2_SEQ) run on across runs; the collective in flight resets with rst. A run stopped
// inside a collective, or aborted, needs X2_CTRL LRST on both cards before the next.
module otpu_coll_xc
  import otpu_pkg::*;
#(
  parameter int S   = 2,
  parameter int NC  = 2,
  parameter int PKT = 32,
  parameter int TXC = 16,
  parameter int RXC = 16,
  parameter int TQ  = 16                 // TX queue words a port
) (
  input  logic              clk,
  input  logic              rst,         // the run's (RUN 0: otpu_board core_rst)
  input  logic              lrst,        // the links' (the board's reset or X2_CTRL LRST)
  input  logic [7:0]        cid,
  input  logic [7:0]        ncards,
  input  logic [31:0]       wdog,
  input  logic              go,          // the card's slices issued a collective (the unit takes it)
  input  cmd_t              cmd,         // its command (slice 0's; held until done)
  input  logic              done,        // the slices' ack (pulse)
  output logic              act,
  output logic              rdone,
  input  logic              tv,          // a local write goes to the write stage: its slice, words
  input  logic [$clog2(S)-1:0] tk,
  input  logic [255:0]      td,
  output logic [S-1:0]      tok,
  output logic              wv,          // another card's write: lanes, addresses, words
  output logic [7:0]        wm,
  output logic [7:0][31:0]  wa,
  output logic [7:0][31:0]  wd,
  input  logic              wt,          //   taken
  output logic [1:0]        l_tv,
  output logic [1:0]        l_tlast,
  output logic [1:0][255:0] l_td,
  output logic [1:0]        l_rcred,
  output logic              l_abort,
  input  logic [1:0]        l_tcred,
  input  logic [1:0]        l_rv,
  input  logic [1:0]        l_rhdr,
  input  logic [1:0]        l_rlast,
  input  logic [1:0][255:0] l_rd,
  input  logic [1:0]        l_up,
  input  logic [1:0]        l_rerr,
  input  logic [1:0]        l_abin,
  output logic [7:0]        xst,
  output logic              xerr,
  output logic [15:0]       xseq         // X2_SEQ: the next collective's number
);
  initial if (PKT < 1 || PKT > 63 || TQ < 4 || (TQ & (TQ - 1)) != 0 || (RXC & (RXC - 1)) != 0 ||
              RXC < 4 || TXC < 1 || 2 * NC > 16)
    $fatal(1, "otpu_coll_xc: PKT %0d (1 .. 63), TQ %0d / RXC %0d (powers of two >= 4), NC %0d (<= 8)",
           PKT, TQ, RXC, NC);
  localparam int QW = $clog2(TQ), RW = $clog2(RXC), CRW = $clog2(TXC + 1);

  // ---- the collective in flight, its shape (from go), the errors
  logic        isbar, xab, e_wd, e_fmt, e_rx, e_ab;
  logic [15:0] seq, cur;                 // the next collective's number, the one in flight's
  logic [15:0] c_rows, c_cols;
  logic [31:0] c_dst, c_drs, c_seg, nb;  // nb: a piece's words (rows x its LANES-word columns)
  logic [4:0]  cnt, ecnt;                // other cards' pieces / tokens in; expected
  logic [15:0] pseen;                    // the pieces in (by g), the tokens (by card)
  logic [31:0] wcnt;
  // each port's parser this cycle: a header out of place; a piece's first header taken (its g);
  // a token taken (its card); a piece's last word taken
  logic [1:0]       ferr, pnew, tdone, pdone;
  logic [1:0][7:0]  pg, psrc;
  logic             wwait, wprog;        // the watchdog: waiting on the links; progress
  wire         txon = ncards > 8'd1 && !xab;
  wire         gbar = cmd.op == OP_BAR || cmd.w3[15:0] == 0 || cmd.w3[31:16] == 0;
  wire         wfire = wwait && wdog != 0 && wcnt == wdog;
  wire  [4:0]  m1 = 5'(ncards) - 5'd1;   // the other cards
  // the hops of slice k's piece out port p (k 0: a token's)
  function automatic logic [3:0] hops(input int k, input int p);
    logic [4:0] hb, hs;                  // the larger half, the smaller
    hb = (m1 + 5'd1) >> 1;
    hs = m1 >> 1;
    return 4'((k % 2 == p) ? hb : hs);
  endfunction

  // the links' side (lrst, never the run's reset: a card whose run starts later finds the other
  // card's words waiting, and the numbers run on across runs): the collectives' numbers, the abort
  always_ff @(posedge clk) begin
    if (lrst) begin
      seq <= '0;
      cur <= '0;
      {xab, e_wd, e_fmt, e_rx, e_ab} <= '0;
      l_abort <= 1'b0;
    end else begin
      l_abort <= !xab && (|ferr || wfire || |l_rerr || |l_abin);   // (on round the ring)
      if (|ferr || wfire || |l_abin || |l_rerr) xab <= 1'b1;
      if (|ferr) e_fmt <= 1'b1;
      if (wfire) e_wd <= 1'b1;
      if (|l_rerr) e_rx <= 1'b1;
      if (|l_abin) e_ab <= 1'b1;
      if (go) begin
        cur <= seq;
        seq <= seq + 1'b1;
      end
    end
  end
  // the collective in flight (the run's)
  always_comb begin
    wwait = act && !xab && (!rdone || tok != '1);
    wprog = |l_rcred || |l_tv || tv;
  end
  always_ff @(posedge clk) begin
    nb <= 32'(c_rows) * 32'((c_cols + 16'd7) >> 3);
    if (go) begin
      isbar <= gbar;
      c_dst <= cmd.w2;
      c_rows <= cmd.w3[15:0];
      c_cols <= cmd.w3[31:16];
      c_drs <= cmd.w5;
      c_seg <= cmd.w6;
    end
    if (rst) begin
      act <= 1'b0;
      rdone <= 1'b0;
      cnt <= '0;
      ecnt <= '0;
      pseen <= '0;
      wcnt <= '0;
    end else if (go) begin
      act <= 1'b1;
      cnt <= '0;
      ecnt <= ncards <= 8'd1 ? 5'd0 : gbar ? m1 : 5'(int'(m1) * S);
      rdone <= ncards <= 8'd1 || xab;
      pseen <= '0;
      wcnt <= '0;
    end else begin
      if (done) act <= 1'b0;
      if (act) rdone <= rdone || xab || cnt == ecnt;
      wcnt <= (!wwait || wprog) ? '0 : wcnt + 1;
      cnt <= cnt + 5'(pdone[0]) + 5'(pdone[1]) + 5'(tdone[0]) + 5'(tdone[1]);
      for (int p = 0; p < 2; p++) begin
        if (pnew[p]) pseen[pg[p][3:0]] <= 1'b1;
        if (tdone[p]) pseen[psrc[p][3:0]] <= 1'b1;
      end
    end
  end

  // ---- TX: the local writes' piece positions (the piece's row and column, its words sent) and
  // the message's word (tm of tl)
  typedef struct packed {
    logic         first, last, bar;
    logic [3:0]   hops;
    logic [15:0]  seq;
    logic [7:0]   gs;
    logic [15:0]  row, col;
    logic [5:0]   len;
    logic [255:0] d;
  } tqe_t;
  logic [15:0] tr, tc;
  logic [31:0] tsent;
  logic [5:0]  tm, tlen;
  logic [5:0]  tl;
  logic        tfirst;
  tqe_t        te, be;                   // the entry a write pushes, a token's
  always_comb begin
    logic [31:0] left;
    left = nb - tsent;
    tfirst = tm == '0;
    tl = tfirst ? ((left >= 32'(PKT)) ? 6'(PKT) : left[5:0]) : tlen;
    te.first = tfirst;
    te.last = tm + 6'd1 == tl;
    te.bar = 1'b0;
    te.hops = '0;                        // (per port)
    te.seq = cur;
    te.gs = 8'(S * int'(cid) + int'(tk));
    te.row = tr;
    te.col = tc;
    te.len = tl;
    te.d = td;
    be = '0;
    be.first = 1'b1;
    be.last = 1'b1;
    be.bar = 1'b1;
    be.seq = seq;                        // (at go: the new collective's)
    be.gs = 8'(S * int'(cid));
  end
  always_ff @(posedge clk) begin
    if (rst || go) begin
      tr <= '0; tc <= '0; tsent <= '0; tm <= '0; tlen <= '0;
    end else if (tv) begin
      tlen <= tl;
      tm <= (tm + 6'd1 == tl) ? 6'd0 : tm + 6'd1;
      if (tsent + 1 == nb) begin         // the piece's last word: the next slice's piece next
        tsent <= '0; tr <= '0; tc <= '0; tm <= '0;
      end else begin
        tsent <= tsent + 1;
        if (32'(tc) + 8 >= 32'(c_cols)) begin
          tc <= '0;
          tr <= tr + 1'b1;
        end else tc <= tc + 16'd8;
      end
    end
  end
`ifndef SYNTHESIS
  // the unit's order: a piece's words, then the next slice's; nb is ready (go + 2) long before
  // the first write (a read's round trip)
  int ck_k, ck_go;
  always @(posedge clk)
    if (rst || go) begin ck_k = 0; ck_go = 0; end
    else begin
      ck_go++;
      if (tv) begin
        if (int'(tk) != ck_k || ck_go < 3)
          $fatal(1, "otpu_coll_xc: a write of slice %0d, piece %0d expected (%0d cycles after go)",
                 tk, ck_k, ck_go);
        if (tsent + 1 == nb) ck_k++;
      end
    end
`endif

  logic [1:0][QW:0] tqn;                 // each port's TX queue words
  for (genvar p = 0; p < 2; p++) begin : g_t
    tqe_t            q [TQ];
    logic [QW-1:0]   wp, rp;
    logic [QW:0]     n;
    logic [CRW-1:0]  cr;                 // the adapter's words free
    logic            hsent;              // the head's header went
    logic [5:0]      orem;               // the message on the link: its words still to go
    logic            push, pop, send, olast;
    tqe_t            e, h;
    logic [255:0]    hw, od;
    always_comb begin
      push = 1'b0;
      e = te;
      if (go && gbar && txon && hops(0, p) != '0) begin
        push = 1'b1;
        e = be;
        e.hops = hops(0, p);
      end else if (tv && txon && hops(int'(tk), p) != '0) begin
        push = 1'b1;
        e.hops = hops(int'(tk), p);
      end
    end
    assign h = q[rp];
    always_comb begin
      hw = '0;
      hw[LKH_TYPE +: 8] = LK_DATA;
      hw[LKH_LEN +: 16] = 16'(h.len);
      hw[LKH_SEQ +: 16] = h.seq;
      hw[LKH_KIND +: 8] = h.bar ? LK_BAR : LK_PIECE;
      hw[LKH_GS +: 8] = h.gs;
      hw[LKH_SRC +: 8] = cid;
      hw[LKH_HOPS +: 4] = h.hops;
      hw[LKH_ROW +: 16] = h.row;
      hw[LKH_COL +: 16] = h.col;
      if (!xab) begin
        send = n != '0 && cr != '0;
        if (h.first && !hsent) begin     // the message's header first
          od = hw;
          olast = h.bar;
        end else begin
          od = h.d;
          olast = h.last;
        end
        pop = send && (h.bar || hsent || !h.first);
      end else begin                     // aborted: the message on the link ends (its words, or
        send = orem != '0 && cr != '0;   //   zeros: the port sends ABORT only between messages),
        od = (n != '0) ? h.d : '0;       //   the rest is dropped
        olast = orem == 6'd1;
        pop = n != '0 && (orem == '0 || send);
      end
    end
    assign l_tv[p] = send;
    assign l_td[p] = od;
    assign l_tlast[p] = olast;
    assign tqn[p] = n;
    always_ff @(posedge clk) if (push) q[wp] <= e;
    always_ff @(posedge clk) begin
      if (lrst) begin
        wp <= '0;
        rp <= '0;
        n <= '0;
        cr <= CRW'(TXC);
        hsent <= 1'b0;
        orem <= '0;
      end else begin
        if (push) wp <= wp + 1'b1;
        if (pop) rp <= rp + 1'b1;
        n <= n + (QW+1)'(push) - (QW+1)'(pop);
        cr <= cr + CRW'(l_tcred[p]) - CRW'(send);
        if (xab) hsent <= 1'b0;
        else if (send) hsent <= h.first && !hsent && !h.bar;
        if (send) begin
          if (!xab && h.first && !hsent && !h.bar) orem <= h.len;
          else if (orem != '0) orem <= orem - 1'b1;
        end
      end
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!lrst && ((push && n == (QW+1)'(TQ) && !pop) || (l_tcred[p] && cr == CRW'(TXC))))
        $fatal(1, "otpu_coll_xc: port %0d's TX queue full (%0d) or a credit over %0d", p, n, TXC);
`endif
  end
  for (genvar k = 0; k < S; k++) begin : g_tok
    assign tok[k] = !txon || ((hops(k, 0) == '0 || tqn[0] < (QW+1)'(TQ)) &&
                              (hops(k, 1) == '0 || tqn[1] < (QW+1)'(TQ)));
  end

  // ---- RX: each port's FIFO and its parser (R_HDR a header next, R_MUL the piece's base,
  // R_DAT rem words of a message), the piece's walk (g, row, column, the row's base); each
  // port's write candidate (cand: lanes cm, addresses ca, words cd), taken (ctk)
  typedef enum logic [1:0] {R_HDR, R_MUL, R_DAT} rph_t;
  logic [1:0]             cand, ctk, ovr;
  logic [1:0][7:0]        cm;
  logic [1:0][7:0][31:0]  ca, cd;
  for (genvar p = 0; p < 2; p++) begin : g_r
    logic [257:0]  q [RXC];              // {hdr, last, word}
    logic [RW-1:0] wp, rp;
    logic [RW:0]   n;
    logic [257:0]  h;
    rph_t          ph;
    logic [5:0]    rem;
    logic          wpc;                  // a piece's walk under way
    logic [7:0]    wg;
    logic [15:0]   wr, wc;
    logic [31:0]   wrow, mul;
    logic          thdr, newp, fe, td_, cv, pop, wlast, plast;
    logic [7:0]    m;
    logic [7:0][31:0] a;
    logic [15:0]   hseq, hrow, hcol, hlen;
    logic [7:0]    hkind, hgs, hsrc;
    always_comb begin
      h = q[rp];
      hseq = h[LKH_SEQ +: 16];
      hkind = h[LKH_KIND +: 8];
      hlen = h[LKH_LEN +: 16];
      hgs = h[LKH_GS +: 8];
      hsrc = h[LKH_SRC +: 8];
      hrow = h[LKH_ROW +: 16];
      hcol = h[LKH_COL +: 16];
      thdr = 1'b0;
      newp = 1'b0;
      fe = 1'b0;
      td_ = 1'b0;
      cv = 1'b0;
      if (n != '0 && !xab && !rst)
        case (ph)
          R_HDR:
            if (!h[257]) fe = 1'b1;
            else if (act && hseq == cur) begin
              if (hkind == LK_BAR) begin
                if (hlen != '0 || !isbar || int'(hsrc) >= int'(ncards) || hsrc == cid ||
                    pseen[hsrc[3:0]]) fe = 1'b1;
                else begin thdr = 1'b1; td_ = 1'b1; end
              end else if (hkind == LK_PIECE && !isbar && hlen != '0 && hlen <= 16'(PKT)) begin
                if (!wpc) begin
                  // another card's slice, the piece's start, not seen yet this collective
                  if (hrow != '0 || hcol != '0 || int'(hgs) >= S * int'(ncards) ||
                      int'(hgs) / S == int'(cid) || pseen[hgs[3:0]]) fe = 1'b1;
                  else begin thdr = 1'b1; newp = 1'b1; end
                end else if (hgs != wg || hrow != wr || hcol != wc) fe = 1'b1;
                else thdr = 1'b1;
              end else fe = 1'b1;
            end else if (hseq != seq) fe = 1'b1;    // (n + 1's: held)
          R_DAT: if (h[257]) fe = 1'b1; else cv = 1'b1;
          default: ;
        endcase
      wlast = 32'(wc) + 8 >= 32'(c_cols);
      plast = wlast && 32'(wr) + 1 == 32'(c_rows);
      for (int l = 0; l < 8; l++) begin
        m[l] = 32'(wc) + 32'(l) < 32'(c_cols);
        a[l] = wrow + 32'(wc) + 32'(l);
      end
    end
    assign pop = !rst && ((n != '0 && xab) || thdr || (cv && ctk[p]));
    assign ferr[p] = fe;
    assign pnew[p] = thdr && newp;
    assign pg[p] = hgs;
    assign tdone[p] = td_;
    assign psrc[p] = hsrc;
    assign cand[p] = cv;
    assign cm[p] = m;
    assign ca[p] = a;
    assign cd[p] = h[255:0];
    assign pdone[p] = cv && ctk[p] && plast;
    assign ovr[p] = cv && ctk[p] && plast && rem != 6'd1;
    assign l_rcred[p] = pop;
    always_ff @(posedge clk) if (l_rv[p]) q[wp] <= {l_rhdr[p], l_rlast[p], l_rd[p]};
    always_ff @(posedge clk) begin
      if (lrst) begin
        wp <= '0;
        rp <= '0;
        n <= '0;
        ph <= R_HDR;
        wpc <= 1'b0;
      end else begin
        if (l_rv[p]) wp <= wp + 1'b1;
        if (pop) rp <= rp + 1'b1;
        n <= n + (RW+1)'(l_rv[p]) - (RW+1)'(pop);
        case (ph)
          R_HDR: if (thdr && hkind == LK_PIECE) begin
            rem <= 6'(hlen);
            if (newp) begin
              wg <= hgs;
              wr <= '0;
              wc <= '0;
              wpc <= 1'b1;
              mul <= 32'(hgs[3:0]) * c_seg;
              ph <= R_MUL;
            end else ph <= R_DAT;
          end
          R_MUL: begin                    // the piece's base, dst + g seg
            wrow <= c_dst + mul;
            ph <= R_DAT;
          end
          R_DAT: if (cv && ctk[p]) begin
            rem <= rem - 1'b1;
            if (rem == 6'd1) ph <= R_HDR;
            if (wlast) begin
              wc <= '0;
              wr <= wr + 1'b1;
              wrow <= wrow + c_drs;
            end else wc <= wc + 16'd8;
            if (plast) wpc <= 1'b0;
          end
          default: ph <= R_HDR;
        endcase
        if (go) wpc <= 1'b0;
        if (xab) ph <= R_HDR;
      end
    end
  end
  // the write candidate: the ports in turn
  logic rr, s1;
  assign s1 = cand[1] && (!cand[0] || rr);
  assign wv = |cand;
  assign wm = cm[s1];
  assign wa = ca[s1];
  assign wd = cd[s1];
  assign ctk = {wt && s1, wt && !s1};
  always_ff @(posedge clk)
    if (lrst) rr <= 1'b0;
    else if (wt) rr <= !s1;

  assign xst = {1'b0, l_up, e_ab, e_rx, e_fmt, e_wd, xab};
  assign xerr = xab || e_rx;
  assign xseq = seq;
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!lrst && |ovr) $fatal(1, "otpu_coll_xc: a message runs past its piece (ports %b)", ovr);
  always @(posedge clk)
    if (!lrst && |ferr && !$test$plusargs("coll_xerr_ok"))
      $fatal(1, "otpu_coll_xc: a header out of place on ports %b (collective %0d, act %0d)", ferr,
             cur, act);
`endif
endmodule

// A card link's crossing (otpu_board NC > 1): the hub's side of one link port (otpu_coll_xc's
// l_* for it) XRS register stages from the port's adapter, which sits beside qsfp-link's lt_port
// in the links' SLR and speaks AXI-Stream to it: the TX words into a FIFO of TXC (a credit back
// to the hub as each goes to the port), the RX words up to the hub only against its credits (RXC:
// the hub's FIFO), so no handshake crosses the dies; the link's status up with them, ABORT and
// CARD_ID down. rst is the links' reset (the hub's lrst: the board's or X2_CTRL LRST, never the
// run's), and the adapter's comes down the stages with the words (as a slice's port's, otpu_board
// g_x), so its counts restart with the hub's links' side.
module otpu_coll_lx
  import otpu_pkg::*;
#(
  parameter int XRS = 2,
  parameter int TXC = 16,
  parameter int RXC = 16
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         ren,            // RUN released: receive words may leave the link
  // the hub's side
  input  logic         t_v,
  input  logic         t_last,
  input  logic [255:0] t_d,
  input  logic         r_cred,
  input  logic         abort,
  input  logic [7:0]   card,
  output logic         t_cred,
  output logic         r_v,
  output logic         r_hdr,
  output logic         r_last,
  output logic [255:0] r_d,
  output logic         up,
  output logic         r_err,
  output logic         ab_in,
  // the port's (lt_port, AXI-Stream on the core clock)
  input  lk_in_t       li,
  output lk_out_t      lo
);
  initial if (XRS < 2 || TXC < 2 || RXC < 2)
    $fatal(1, "otpu_coll_lx: XRS %0d (>= 2), TXC %0d, RXC %0d (>= 2)", XRS, TXC, RXC);
  localparam int DW = 1 + 3 + 1 + 1 + 256 + 8;   // pulses (t_v, r_cred, abort), rst, t_last, t_d, card
  localparam int UW = 4 + 1 + 1 + 1 + 256;   // pulses (t_cred, r_v, r_err, ab_in), up, r_hdr, r_last, r_d
  localparam int TW = $clog2(TXC), CW = $clog2(RXC + 1);
  logic [DW-1:0] dn_d, dn_q;
  logic [UW-1:0] up_d, up_q;
  logic [XRS-2:0][DW-1:0] dn_m;
  logic [XRS-2:0][UW-1:0] up_m;
  logic          a_rst, a_ren, a_tv, a_tlast, a_rcred, a_abort;
  logic [255:0]  a_td;
  logic [7:0]    a_card;
  assign dn_d = {ren, card, t_d, t_last, rst, abort, r_cred, t_v};
  (* keep_hierarchy = "yes" *) otpu_xr #(.W(DW), .RW(3)) u_dn_tx (.clk, .rst, .d(dn_d), .q(dn_m[0]));
  for (genvar i = 1; i < XRS - 1; i++) begin : g_dm
    otpu_xr #(.W(DW), .RW(3)) u_dn (.clk, .rst, .d(dn_m[i-1]), .q(dn_m[i]));
  end
  (* keep_hierarchy = "yes" *) otpu_xr #(.W(DW), .RW(3)) u_dn_rx (.clk, .rst(a_rst), .d(dn_m[XRS-2]), .q(dn_q));
  assign {a_ren, a_card, a_td, a_tlast, a_rst, a_abort, a_rcred, a_tv} = dn_q;

  // the adapter: TX FIFO (TXC; a credit up as its head goes), RX credits (RXC)
  logic [256:0]  tq [TXC];
  logic [TW-1:0] twp, trp;
  logic [TW:0]   tn;
  logic [CW-1:0] rc;
  logic          tpop, rtake;
  assign tpop = tn != '0 && li.s_tready;
  assign lo.s_tvalid = tn != '0;
  assign lo.s_tdata = tq[trp][255:0];
  assign lo.s_tlast = tq[trp][256];
  assign lo.m_tready = !a_rst && a_ren && rc != '0;
  assign lo.abort_req = a_abort;
  assign lo.card = a_card;
  assign rtake = li.m_tvalid && lo.m_tready;
  always_ff @(posedge clk) if (a_tv) tq[twp] <= {a_tlast, a_td};
  always_ff @(posedge clk) begin
    if (a_rst) begin
      twp <= '0;
      trp <= '0;
      tn <= '0;
      rc <= CW'(RXC);
    end else begin
      if (a_tv) twp <= (twp == TW'(TXC - 1)) ? '0 : twp + 1'b1;
      if (tpop) trp <= (trp == TW'(TXC - 1)) ? '0 : trp + 1'b1;
      tn <= tn + (TW+1)'(a_tv) - (TW+1)'(tpop);
      rc <= rc + CW'(a_rcred) - CW'(rtake);
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!a_rst && ((a_tv && tn == (TW+1)'(TXC) && !tpop) || (a_rcred && rc == CW'(RXC))))
      $fatal(1, "otpu_coll_lx: a TX word into a full FIFO (%0d) or an RX credit over %0d", TXC, RXC);
`endif

  assign up_d = {li.m_tdata, li.m_tlast, li.m_thdr, li.up, li.abort_in, li.rx_err, rtake, tpop};
  (* keep_hierarchy = "yes" *) otpu_xr #(.W(UW), .RW(4)) u_up_tx (.clk, .rst(a_rst), .d(up_d), .q(up_m[0]));
  for (genvar i = 1; i < XRS - 1; i++) begin : g_um
    otpu_xr #(.W(UW), .RW(4)) u_up (.clk, .rst(a_rst), .d(up_m[i-1]), .q(up_m[i]));
  end
  (* keep_hierarchy = "yes" *) otpu_xr #(.W(UW), .RW(4)) u_up_rx (.clk, .rst, .d(up_m[XRS-2]), .q(up_q));
  assign {r_d, r_last, r_hdr, up, ab_in, r_err, r_v, t_cred} = up_q;
endmodule

// A slice's collective port at S > 1 (otpu_board): the hub's reads and writes (a FIFO of F)
// presented on the slice's collective TMEM port and granted by the slice's own TMEM arbiter (gnt
// = gnt_local: the collective group's grant, reads and writes together), the read data returned
// the cycle after it is fresh, each written entry returned as a pop. NR 1: one read at a time,
// held (hv; hq: hv's second cycle. The held read is asked from it (TMEM's input stage, release r2,
// docs/tmem_nocopy.md 7: every unit's read goes to TMEM a cycle after it forms, the writes a
// cycle after their grant), and held there until granted, with no replay). NR > 1 (the hub's
// reads in flight): the reads queue in a FIFO of NR (g_pipe: qm, qa), and the oldest is asked
// from flip-flops (hv, hm, ha: loaded the cycle after the read is in the FIFO, so asked two cycles
// after it came at the earliest, as hq; held there until granted, with no replay), the next one
// the cycle after a grant: a read a granted cycle.
module otpu_coll_port #(
  parameter int LANES = 8,
  parameter int F     = 8,
  parameter int NR    = 1,
  parameter bit WT    = 1'b0              // the writes' FOLLOW tag (otpu_board NC > 1; 0: coll_wtop 1)
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
  input  logic                    wt,
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
  output logic                    coll_wtop,   // FOLLOW: the write advances the top (otpu_slice)
  input  logic                    gnt_local,
  output logic                    gnt
);
  localparam int PW = $clog2(F);         // F a power of two: the pointers wrap
  logic                   hv, got;
  logic [LANES-1:0]       hm;
  logic [LANES-1:0][31:0] ha;
  logic [LANES-1:0]       fm [F];
  logic [LANES-1:0][31:0] fa [F];
  logic [LANES-1:0][31:0] fd [F];
  logic [PW-1:0]          wp, rp;
  logic [PW:0]            n;
  wire                    wtake = n != '0 && gnt_local;

  assign gnt = gnt_local;
  assign coll_raddr = ha;
  assign coll_wen = n != '0 ? fm[rp] : '0;
  assign coll_waddr = fa[rp];
  assign coll_wdata = fd[rp];
  if (WT) begin : g_wt
    logic ft [F];
    always_ff @(posedge clk) if (wv) ft[wp] <= wt;
    assign coll_wtop = ft[rp];
  end else begin : g_nwt
    assign coll_wtop = 1'b1;
  end

  always_ff @(posedge clk) begin
    if (got) dd <= coll_rdata;          // fresh the cycle after the granted read
    if (wv) begin
      fm[wp] <= wm;
      fa[wp] <= wa;
      fd[wp] <= wd;
    end
    if (rst) begin
      dv <= 1'b0;
      pop <= 1'b0;
      wp <= '0;
      rp <= '0;
      n <= '0;
    end else begin
      dv <= got;
      pop <= wtake;
      if (wv) wp <= wp + 1'b1;
      if (wtake) rp <= rp + 1'b1;
      n <= n + (PW+1)'(wv) - (PW+1)'(wtake);
    end
  end

  if (NR == 1) begin : g_one
    logic hq;
    assign coll_ren = (hv && hq) ? hm : '0;
    always_ff @(posedge clk) begin
      if (rv) begin
        hm <= rm;
        ha <= ra;
      end
      if (rst) begin
        hv <= 1'b0;
        hq <= 1'b0;
        got <= 1'b0;
      end else begin
        got <= hv && hq && gnt_local;
        if (rv) hv <= 1'b1;
        else if (hv && hq && gnt_local) hv <= 1'b0;
        hq <= hv && !(hq && gnt_local);
      end
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && rv && hv) $fatal(1, "otpu_coll_port: a read while one is held");
`endif
  end else begin : g_pipe
    localparam int RW = $clog2(NR);
    logic [LANES-1:0]       qm [NR];
    logic [LANES-1:0][31:0] qa [NR];
    logic [RW-1:0]          qw, qr;
    logic [RW:0]            qn;
    wire                    take = hv && gnt_local;          // the oldest read granted
    wire                    load = qn != '0 && (!hv || take); // the next one to the flip-flops
    assign coll_ren = hv ? hm : '0;
    always_ff @(posedge clk) if (rv) qm[qw] <= rm;
    always_ff @(posedge clk) if (rv) qa[qw] <= ra;
    always_ff @(posedge clk) begin
      if (load) begin
        hm <= qm[qr];
        ha <= qa[qr];
      end
      if (rst) begin
        hv <= 1'b0;
        got <= 1'b0;
        qw <= '0;
        qr <= '0;
        qn <= '0;
      end else begin
        got <= take;
        hv <= load || (hv && !take);
        if (rv) qw <= qw + 1'b1;
        if (load) qr <= qr + 1'b1;
        qn <= qn + (RW+1)'(rv) - (RW+1)'(load);
      end
    end
`ifndef SYNTHESIS
    always @(posedge clk)
      if (!rst && rv && qn == (RW+1)'(NR) && !load)
        $fatal(1, "otpu_coll_port: a read into a full FIFO (%0d)", NR);
`endif
  end
`ifndef SYNTHESIS
  always @(posedge clk)
    if (!rst && wv && n == (PW+1)'(F) && !wtake)
      $fatal(1, "otpu_coll_port: a write into a full FIFO");
`endif
endmodule

// One register stage of an SLR crossing (otpu_board at S > 1: XRS per direction and slice, the
// first and the last on each side). The low RW bits (valid pulses, request levels) clear with rst; the others
// are data. max_fanout: a stage's reset bit drives a slice's whole core reset (as core_rst).
module otpu_xr #(
  parameter int W  = 1,
  parameter int RW = 0
) (
  input  logic         clk,
  input  logic         rst,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  (* max_fanout = 256 *) logic [W-1:0] r;     // r_reg: the floorplan's USER_SLL_REG cells
  assign q = r;
  always_ff @(posedge clk) begin
    r <= d;
    if (rst) for (int i = 0; i < RW; i++) r[i] <= 1'b0;
  end
endmodule
