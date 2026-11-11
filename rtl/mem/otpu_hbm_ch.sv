// One memory channel's bridge: the accelerator's native master (otpu_native_dram's per-channel
// port, core clock, one CB-byte beat per command) onto one AXI3 port of a memory controller
// (256-bit, aclk: the BCU1525's WG builds at 300 MHz, behind otpu_mig_pair). It was written for a
// stacked memory's (HBM) pseudo-channels, whose names it keeps: a build has one a channel, PC
// its number (channel beat m is at address PC x 2^(CAW + log2 CB) + m x CB). The AXI address
// is OAW bits: {PC, m, log2(CB) zeros} cut to OAW, so a channel of more than 512 MiB at OAW 34
// (the BCU1525's: CAW 25, 4 GiB) leaves PC fewer bits, which must hold it (PC 0 there).
//
// The native port (core clock) has otpu_mem_ch's semantics: a command stream (we, beat); one
// write-data beat (data, byte mask with 1 = write the byte) per write command, in write-command
// order, before or after its command; read data in read-command order with no backpressure (the
// master reserves room for its reads); n_wdone, the write beats the memory has taken, mod 2^16:
// here a beat is counted once its burst's B has come back, so a write counted in n_wdone is
// visible to every other AXI master of the memory (the host's path beside the bridges).
// One master's commands execute in command order: a read issued after a write to its beat returns
// the written data without waiting for n_wdone, and a write issued after a read of its beat does
// not change what the read returns.
//
// Posted reads (n_csrc 3: port W's, nch's XPOST; docs/bcu1525_wg.md): a read
// command carries its source (n_csrc: 0 B, 1 A, 2 SW fill, 3 posted) and, posted, its chunk FIFO
// entry (n_cslot). A posted read's beat never comes on n_rvalid: its K AXI beats go out on p_*
// (aclk, no ready: the MXU reserved the entry) as quarter writes into entry p_slot of the
// channel's lane RAM, in the channel's command order, and p_cntg, the gray code of the whole
// beats posted (mod 2^(XSW + 2), from a register, never reset), steps after a beat's last quarter.
//
// The bridge is in two halves joined by a register pipeline (the (B) split): otpu_hbm_cs, the
// core half, beside the slice (the native port, the clock crossings, the RAMs at the native
// width, the command and W pumps, the R sink and the posted writes); otpu_hbm_ac, the AXI half,
// beside the HBM port (the gather, the issue, the hazards, the read credits, the W FIFO and its
// engine, B, the drain and the hold, the response checks); otpu_hbm_ln between them, N + 1 plain
// register stages each way (N 0 for a bridge at its port; a far slice's, across the SLRs, as
// RLY: stage 0 beside the core half). Everything that decides on the AXI side is the AXI half's;
// what crosses is a command a cycle at most (26 signals), W (a half AXI beat a cycle at WH, else
// a whole one), R (an AXI beat a cycle), the credits of the AXI half's two FIFOs and of u_rd back,
// a pulse per write beat answered, the error bit, the hold and the core's reset request: 436
// signals a channel at WH.
//
// The commands cross into aclk through an asynchronous FIFO (u_cq), the write data through
// another (u_wd); the read data comes back through K more (g_rd[q].u_rd: AXI beat q of each native
// beat, written a quarter at a time at aclk, read together at the core clock).
//
// Bursts (the AXI half). A register (the gather, g_*) collects consecutive commands of one kind
// and source on consecutive beats of one aligned block of MAXB = 16 / K beats (512 bytes: a burst
// never crosses a 4 KiB boundary) into one burst of up to 16 AXI beats; whether a command
// continues the one before it is decided at the core clock (c_seq, carried with the command). The
// burst is complete when the block is, when the next command does not continue it, or after mwait
// aclk cycles without a next command (a sequential stream at the core clock arrives one beat every
// 3.4 aclk cycles at 133.33 MHz; mwait 0: at once); then it moves into the issue register (i_*)
// once that is free, and goes out from there when it has room and no hazard holds it. While the
// issue register waits the gather goes on collecting. What an issue changes (the bursts-in-flight
// FIFOs, the credits, the W beats claimed, the hazard counts) follows a cycle later (p_*), and the
// issue's test is registered (i_ok): the issue register reloads in the cycle after an issue at the
// earliest and tests the new burst in the cycle after that, with those changes in, so bursts go
// three cycles apart at the least (on the port a burst of one 128-byte beat takes 4, of 4 16). (At
// 450 MHz the issue's decision and its effects in one cycle took 6-8 logic levels: probe build 3
// placed, -1.1 to -2.3 ns.)
//
// Reads. A read burst of sources 0-2 goes when u_rd has room for MAXB beats beyond every such
// read issued and not yet freed (rcr: in flight, in the pipeline or in u_rd; the core half sends a
// pulse up the pipeline for each u_rd entry the core clock pops), with up to ARD beats out: 14.4
// GB/s x ~300 ns of HBM read latency is ~34 beats of 128 bytes. A posted burst takes no u_rd room
// (the MXU reserved its entries); the reads in flight are bounded by the core half's record FIFO
// (u_rc: a read's posted flag and slot, RCD entries; the command pump sends no command without
// room in it). One ARID (0), so the HBM returns the bursts in order; the R beats are always taken (rready
// 1) and go down the pipeline, where the core half's sink writes each into its u_rd quarter or
// posts it, the record FIFO's head saying which; a burst ends at RLAST (its beat count checked
// against RLAST a cycle later, an n_err).
//
// Write data, core side: n_wvalid / n_wready, a handshake (WPOST 0), or (WPOST 1) a posted
// strobe on the writer's own credit, n_wfree: u_wd's room as the core side sees it, WDD less what
// it wrote (a strobe counted from the cycle after it) less what the aclk side popped as last seen
// through the synchronizer (so never more than there is), from registers only, and 0 while the
// core side is in reset (a_crst). A posted write into a full u_wd is a simulation $error, except in
// a hold (a_hs2, a_crst: u_wd's write side in reset, and its room as of the read side's reset, the
// cycle before, not yet the write side's; the hold drops what u_wd holds). The data may come ahead
// of its write command (the adapter's takes, nch's g2), in write-command order.
//
// Writes. The core half's W pump sends u_wd's head up the pipeline as its K AXI beats (WH: each in
// two halves, 128 data bits and 16 strobes, on consecutive cycles) while the AXI half's W FIFO
// (AWD AXI beats, the halves joined) has room (a credit an entry, back at its pops). A write burst
// goes only once its data is all in the W FIFO (the beats there not yet given to an issued burst:
// wav), so that its W beats follow its AW without gaps: the HBM's AXI port 0 is shared with the
// host's writes (W in AW order, no WID), which must never wait on the core. One AWID (0): the
// bursts complete in order. A W engine sends each issued burst's beats from the W FIFO's head,
// the native beat's byte mask as WSTRB (the HBM takes byte masks: no read-modify-write). B always
// taken (bready 1).
//
// Ordering. AXI does not order reads against writes, even with one ID. So a read burst waits while
// a write burst to its bucket (a hash of its 512-byte block) is in flight (issued, no B yet), and a
// write burst waits while a read burst to its bucket is in flight (issued, last beat not back):
// per bucket a count of each (rcnt, wcnt) and its nonzero flag; a count takes a burst's end a
// cycle late (q_*), so its hazard holds a cycle longer. A collision only delays. Since the issue
// register issues in command order, every earlier command is in flight or done when a burst goes.
// With BHW (the bank-half window, otpu_mig_pair's) a read from the window's upper half up neither
// waits nor counts: a run writes below it alone (the weights above it: the images' layout, and
// the ISA simulator's guard on every ST, DSTEP and QST, Slice.wlim), so nothing the core writes is
// there to order it against; by address, not by source (attention's KV reads are posted too, and
// the KV is written).
//
// n_wdone: B adds its burst's beats to a count of beats answered (wd_pend), which leaves it one
// beat per aclk cycle as a pulse down the pipeline into the core half's count, gray-coded across
// into the core clock (one bit per source cycle, as otpu_mem_ch's).
//
// Resets. arst (the HBM's AXI reset; the HBM's ports reset with it) clears the AXI half at once
// and holds it (a_hold, at least max(16, 2 (N + 1) + 4) cycles: both halves held while the
// pipeline empties both ways, the core half's hold a register past it); the hold goes down the
// pipeline (its stages start at 1) and through that register resets the core half's aclk side,
// and into the core clock (a_hs). The core's rst is a request (a_req, kept up until the hold is
// seen back), synchronized in the core half, then up the pipeline: the core half's pumps stop
// and its sink writes nothing more (no u_rd entry, no posted beat, no count: a late count
// completes an MXU entry before its data); the AXI half stops issuing and drops the gather, lets
// what is in flight finish (the W engine sends the bursts it has issued from its W FIFO;
// returning reads go down to the sink, which drops them; B's come back), then holds. The core
// side stays in reset (a_crst) until the hold, synchronized, is over. So the HBM never sees a
// burst without its W beats, nor returns data or a B the bridge does not expect.
//
// n_err (sticky until arst; core clock): R or B without a burst in flight, RLAST off the burst's
// last beat, RRESP or BRESP not OKAY.
//
// ev (aclk, pulses, for the probe's counters): {a write burst held by a hazard, a read burst held
// by credits, an AW issued, an AR issued}; ev_n: the issue register's burst's native beats.
//
// Half beats (RHB, WHB: the WG targets' R1 / R2 with otpu_native_dram's RHB / WHB, docs/native_channels.md;
// CB 128; 0, the default: whole beats). RHB: n_rdata carries half a beat a cycle (its low CB x 4
// bits, the rest 0; n_rvalid each half, the low one first), and u_rd is two FIFOs of K / 2 lanes'
// AXI beats each (g_rd[l]: lane l holds AXI beats l and l + K / 2 of each native beat, 2 ARD
// entries), popped together: each pop is the next half beat, with no selection. A beat's low half
// goes only with the whole beat in u_rd and no reset up (the core's, the hold), its high half the
// next cycle whatever the resets. The read credits stay in native beats (a free when lane 1 has
// freed two entries: the beat's last quarter popped). WHB (needs WPOST): the write
// data comes in halves (n_wdata's low CB x 4 bits, n_wmask's low CB / 2; n_wvalid each half, in
// order), and u_wd holds half beats (CB x 9 / 2 bits and the half's flag, WDD entries; the W pump
// fails an entry of the wrong half, a simulation $error): n_wfree counts its whole beats' room (half
// the free entries, rounded down), and the W pump sends an entry as K / 2 AXI beats.
// The AXI half sees the same AXI beats either way.
//
// CPY (the BCU1525's two slices, otpu_bcu_wg_sys at S 2; 0, the default: as above): the core
// half's widest aclk fan-outs in copies of their registers, cycle for cycle the same bridge: the
// hold's FIFO resets (u_cq's and u_wd's read sides, each u_rd's write side) and each u_rd's RAM
// write enable and address (otpu_afifo WCP 4, 64 bits a copy); in the AXI half the W FIFO's head
// address (u_wf, otpu_sfifo HCP 4). s2syn (2026-11-01) placed a slice's core halves over 100-180
// rows of its SLR, beside the DDR4 controller of its die, and these were most of the aclk paths
// that failed at 300 MHz.
//
// Clock-domain crossings (the core half's): every synchronizer carries ASYNC_REG; their
// constraints are in boards/bcu1525/constraints/otpu_hbm_ch.tcl.
module otpu_hbm_ch #(
  parameter int CB  = 128,                 // native beat bytes (64 or 128)
  parameter int CAW = 22,                  // channel beat address bits (512 MiB / CB)
  parameter logic [4:0] PC = 5'd0,         // the pseudo-channel: AXI addresses PC x 2^29 + ...
  parameter int OAW = 34,                  // the AXI address's bits (PC x 2^(CAW + log2(CB)) + ...)
  parameter int ARD = 64,                  // read-data FIFO, native beats (the read credits)
  parameter int CQD = 32,                  // command FIFO (core clock to aclk)
  parameter int WDD = 32,                  // write-data FIFO (RAM32M16's depth; the adapter's
                                           // write data waits here, ahead of its commands)
  parameter int WOT = 16,                  // write bursts in flight (issued, no B yet)
  parameter int NBK = 32,                  // hazard buckets
  parameter bit WPOST = 1'b0,              // n_wvalid a posted strobe on n_wfree's credit
  parameter int N   = 0,                   // the pipeline between the halves: N + 1 stages
  parameter bit WH  = 1'b0,                // W up the pipeline at half width (a beat in 2 cycles)
  parameter int ACQ = 16,                  // the AXI half's command FIFO
  parameter int AWD = 32,                  // the AXI half's W FIFO, AXI beats
  parameter int RCD = 64,                  // the record FIFO: reads in flight
  parameter int PST = 1,                   // p_*'s register stages
  parameter int XSW = 9,                   // a chunk FIFO entry's bits (log2 FIFO_DEPTH)
  parameter bit RHB = 1'b0,                // n_rdata in half beats (see the top)
  parameter bit WHB = 1'b0,                // n_wdata / n_wmask in half beats (WPOST)
  parameter int BHW = 0,                   // the bank-half window: the channel's first 2^BHW bytes
                                           // (0: none); reads from 2^(BHW - 1) up skip the hazards
  parameter bit CPY = 1'b0                 // the core half's aclk fan-outs in copies (otpu_hbm_cs)
) (
  // the accelerator (otpu_native_dram's channel port), core clock
  input  logic            clk,
  input  logic            rst,
  input  logic            n_cvalid,
  output logic            n_cready,
  input  logic            n_cwe,
  input  logic [CAW-1:0]  n_caddr,         // the beat in the channel
  input  logic [1:0]      n_csrc,          // a read's source: 0 B, 1 A, 2 SW fill, 3 posted
  input  logic [XSW-1:0]  n_cslot,         // a posted read's chunk FIFO entry
  input  logic            n_wvalid,        // one beat per write command, in write-command order
  output logic            n_wready,
  output logic [$clog2(WDD):0] n_wfree,    // u_wd's room in beats (WPOST's credit)
  input  logic [CB*8-1:0] n_wdata,
  input  logic [CB-1:0]   n_wmask,         // 1 = write the byte
  output logic            n_rvalid,        // in read-command order, no backpressure (RHB: halves)
  output logic [CB*8-1:0] n_rdata,
  output logic [15:0]     n_wdone,         // write beats answered by the HBM, mod 2^16
  output logic            n_err,           // a response the bridge did not expect (sticky)
  // posted reads' beats (aclk), quarter by quarter, and their count: p_q one-hot (bit q with
  // p_we, else 0) and p_slot in copies of equal registers, one a lane RAM group (otpu_mxu XPOST:
  // p_q the quarter RAMs' 0, the remainder's field enables 1; p_slot quarter RAM g's g, the
  // remainder RAM's 4)
  output logic            p_we,
  output logic [1:0][CB/32-1:0] p_q,
  output logic [4:0][XSW-1:0] p_slot,
  output logic [255:0]    p_d,
  output logic [XSW+1:0]  p_cntg,
  // the HBM's AXI3 port, aclk (ID 0, AxSIZE 32 bytes, INCR: the top ties them)
  input  logic            aclk,
  input  logic            arst,
  input  logic [5:0]      mwait,           // aclk cycles a burst waits for its next beat (static)
  output logic            m_awvalid,
  input  logic            m_awready,
  output logic [OAW-1:0]  m_awaddr,
  output logic [3:0]      m_awlen,
  output logic            m_wvalid,
  input  logic            m_wready,
  output logic [255:0]    m_wdata,
  output logic [31:0]     m_wstrb,
  output logic            m_wlast,
  input  logic            m_bvalid,
  output logic            m_bready,
  input  logic [1:0]      m_bresp,
  output logic            m_arvalid,
  input  logic            m_arready,
  output logic [OAW-1:0]  m_araddr,
  output logic [3:0]      m_arlen,
  input  logic            m_rvalid,
  output logic            m_rready,
  input  logic [255:0]    m_rdata,
  input  logic [1:0]      m_rresp,
  input  logic            m_rlast,
  output logic [3:0]      ev,              // aclk pulses: {haz_w, cred_r, aw, ar}
  output logic [$clog2(16 / (CB / 32)):0] ev_n
);
  localparam int WS = WH ? 145 : 289;      // W up the pipeline: a half or a whole AXI beat, a flag
  localparam int FW = CAW + 4 + 1 + WS + 2;  // core half to AXI half: command, W, a free, the request
  localparam int BW = 257 + 5;             // back: R, the two credits, a write beat, n_err, the hold
  logic           a_crst;                  // the core side in reset (tb_hbmch's master reads it)
  logic           c_v, c_we, c_seq, c_post, w_v, rf, rq;
  logic [CAW-1:0] c_a;
  logic [WS-1:0]  w_d;
  logic           r_v, cc, wc, bp, er, hd;
  logic [255:0]   r_d;
  logic           c_v_a, c_we_a, c_seq_a, c_post_a, w_v_a, rf_a, rq_a;
  logic [CAW-1:0] c_a_a;
  logic [WS-1:0]  w_d_a;
  // CPY: stage N's w_v and w_d[0] (the half's flag) in WC copies, {w_v, w_d[0]} each, for the AXI
  // half's W join (otpu_hbm_ac: its two w_lo halves' enables at WH, the W FIFO's write)
  localparam int WC = CPY ? (WH ? 3 : 1) : 0;
  logic [(WC > 0 ? WC : 1)-1:0][1:0] w_c_a;
  logic           r_v_a, cc_a, wc_a, bp_a, er_a, hd_a;
  logic [255:0]   r_d_a;

  // The halves and the pipeline keep their hierarchy through synthesis (KEEP_HIERARCHY; the
  // pipeline DONT_TOUCH too): the floorplan places them by name and the crossings' constraints
  // are scoped to the core half. Probe build 8's synthesis (flatten_hierarchy rebuilt) put six
  // channels' u_rd / u_wd RAMs under u_mem instead (their crossings then timed as synchronous:
  // -1.29 ns at the core clock) and named some of one channel's stage registers another's.
  (* keep_hierarchy = "yes" *)
  otpu_hbm_cs #(.CB(CB), .CAW(CAW), .PC(PC), .ARD(ARD), .CQD(CQD), .WDD(WDD), .WPOST(WPOST), .WH(WH),
                .ACQ(ACQ), .AWD(AWD), .RCD(RCD), .PST(PST), .XSW(XSW), .RHB(RHB), .WHB(WHB),
                .CPY(CPY)) u_cs (
    .clk, .rst, .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_csrc, .n_cslot, .n_wvalid, .n_wready,
    .n_wfree, .n_wdata, .n_wmask, .n_rvalid, .n_rdata, .n_wdone, .n_err, .o_crst(a_crst),
    .aclk, .p_we, .p_q, .p_slot, .p_d, .p_cntg,
    .c_v, .c_we, .c_seq, .c_post, .c_a, .w_v, .w_d, .rf, .rq,
    .r_v, .r_d, .cc, .wc, .bp, .er, .hd);
  (* keep_hierarchy = "yes", dont_touch = "true" *)
  otpu_hbm_ln #(.N(N), .FW(FW), .BW(BW), .BI(BW'(1)), .FC(WC), .FA(2), .FB(2 + WS)) u_ln (.aclk,
    .f_i({c_v, c_we, c_seq, c_post, c_a, w_v, w_d, rf, rq}),
    .f_o({c_v_a, c_we_a, c_seq_a, c_post_a, c_a_a, w_v_a, w_d_a, rf_a, rq_a}), .f_c(w_c_a),
    .b_i({r_v_a, r_d_a, cc_a, wc_a, bp_a, er_a, hd_a}),
    .b_o({r_v, r_d, cc, wc, bp, er, hd}));
  (* keep_hierarchy = "yes" *)
  otpu_hbm_ac #(.CB(CB), .CAW(CAW), .PC(PC), .OAW(OAW), .ARD(ARD), .WOT(WOT), .NBK(NBK), .N(N),
                .WH(WH), .ACQ(ACQ), .AWD(AWD), .BHW(BHW), .CPY(CPY)) u_ac (
    .aclk, .arst, .mwait,
    .c_v(c_v_a), .c_we(c_we_a), .c_seq(c_seq_a), .c_post(c_post_a), .c_a(c_a_a), .w_v(w_v_a),
    .w_d(w_d_a), .w_c(w_c_a), .rf(rf_a), .rq(rq_a),
    .r_v(r_v_a), .r_d(r_d_a), .cc(cc_a), .wc(wc_a), .bp(bp_a), .er(er_a), .hd(hd_a),
    .m_awvalid, .m_awready, .m_awaddr, .m_awlen, .m_wvalid, .m_wready, .m_wdata, .m_wstrb, .m_wlast,
    .m_bvalid, .m_bready, .m_bresp, .m_arvalid, .m_arready, .m_araddr, .m_arlen,
    .m_rvalid, .m_rready, .m_rdata, .m_rresp, .m_rlast, .ev, .ev_n);
endmodule

// The pipeline between the bridge's halves: N + 1 plain register stages each way (no enable, no
// reset: the hold and the reset request are bits of it), stage 0 beside the core half, stage N
// beside the AXI half. KEEP: synthesis merges registers with the same input (probe build 7's
// relays). The floorplan places them (otpu_hbm_fp.tcl; N = 4 k + pre for a slice in SLR k):
// stages j < pre with the core half, stage j >= pre in SLR
// k - ceil((j - pre) / 4), a stage a clock region row, stage N with the AXI half; the Laguna TX /
// RX pairs (stages 4 b + pre and 4 b + pre + 1) fed and read by registers only.
// BI: the backward bundle's value from configuration (the hold's 1).
module otpu_hbm_ln #(
  parameter int N  = 0,
  parameter int FW = 1,
  parameter int BW = 1,
  parameter logic [BW-1:0] BI = '0,
  parameter int FC = 0,                     // stage N's f bits FA and FB in FC copies (0: none)
  parameter int FA = 0,
  parameter int FB = 0
) (
  input  logic          aclk,
  input  logic [FW-1:0] f_i,                // the core half's, into stage 0
  output logic [FW-1:0] f_o,                // stage N's
  output logic [(FC > 0 ? FC : 1)-1:0][1:0] f_c,   // the copies: {f_o[FB], f_o[FA]} each
  input  logic [BW-1:0] b_i,                // the AXI half's, into stage N
  output logic [BW-1:0] b_o                 // stage 0's
);
  (* shreg_extract = "no", keep = "true" *) logic [N:0][FW-1:0] fs = '0;
  (* shreg_extract = "no", keep = "true" *) logic [N:0][BW-1:0] bs = {(N + 1){BI}};
  always_ff @(posedge aclk) begin
    fs[0] <= f_i;
    bs[N] <= b_i;
    for (int j = 1; j <= N; j++) begin
      fs[j]     <= fs[j - 1];
      bs[j - 1] <= bs[j];
    end
  end
  assign f_o = fs[N];
  assign b_o = bs[0];
  // FC: stage N's two bits again in FC registers each, loaded from stage N - 1 (f_i at N 0) as
  // stage N is (DONT_TOUCH: synthesis merges registers with the same input; not the floorplan's
  // stage cells, fs_reg: the placer puts each beside its loads)
  if (FC > 0) begin : g_fc
    logic [FW-1:0] fp;                      // stage N's input
    assign fp = N == 0 ? f_i : fs[N == 0 ? 0 : N - 1];
    (* shreg_extract = "no", dont_touch = "true" *) logic [FC-1:0][1:0] fc = '0;
    always_ff @(posedge aclk) fc <= {FC{fp[FB], fp[FA]}};
    assign f_c = fc;
`ifndef SYNTHESIS
    always_ff @(posedge aclk)
      if (fc != {FC{fs[N][FB], fs[N][FA]}}) $fatal(1, "otpu_hbm_ln: the copies are not stage N's");
`endif
  end else begin : g_fc0
    assign f_c = '0;
  end
endmodule

// The bridge's core half (otpu_hbm_ch): the native port, the clock crossings (u_cq, u_wd, u_rd),
// the command pump (u_cq's head up the pipeline on the AXI half's command FIFO's credits and, for
// a read, room in the record FIFO), the W pump (u_wd's head as its AXI beats, in halves at WH, on
// the W FIFO's credits), the R sink (each AXI beat into its u_rd quarter or posted, from the
// record FIFO's head), the posted writes and their count, u_rd's frees back to the AXI half's
// read credits, n_wdone's count and n_err into the core clock.
module otpu_hbm_cs #(
  parameter int CB  = 128,
  parameter int CAW = 22,
  parameter logic [4:0] PC = 5'd0,
  parameter int ARD = 64,
  parameter int CQD = 32,
  parameter int WDD = 32,
  parameter bit WPOST = 1'b0,
  parameter bit WH  = 1'b0,
  parameter int ACQ = 16,
  parameter int AWD = 32,
  parameter int RCD = 64,
  parameter int PST = 1,
  parameter int XSW = 9,
  parameter bit RHB = 1'b0,
  parameter bit WHB = 1'b0,
  parameter bit CPY = 1'b0
) (
  input  logic            clk,
  input  logic            rst,
  input  logic            n_cvalid,
  output logic            n_cready,
  input  logic            n_cwe,
  input  logic [CAW-1:0]  n_caddr,
  input  logic [1:0]      n_csrc,
  input  logic [XSW-1:0]  n_cslot,
  input  logic            n_wvalid,
  output logic            n_wready,
  output logic [$clog2(WDD):0] n_wfree,
  input  logic [CB*8-1:0] n_wdata,
  input  logic [CB-1:0]   n_wmask,
  output logic            n_rvalid,
  output logic [CB*8-1:0] n_rdata,
  output logic [15:0]     n_wdone,
  output logic            n_err,
  output logic            o_crst,
  input  logic            aclk,
  output logic            p_we,
  output logic [1:0][CB/32-1:0] p_q,
  output logic [4:0][XSW-1:0] p_slot,
  output logic [255:0]    p_d,
  output logic [XSW+1:0]  p_cntg,
  // up the pipeline (into stage 0)
  output logic            c_v,
  output logic            c_we,
  output logic            c_seq,
  output logic            c_post,
  output logic [CAW-1:0]  c_a,
  output logic            w_v,
  output logic [(WH ? 145 : 289)-1:0] w_d,
  output logic            rf,              // a u_rd entry freed
  output logic            rq,              // the core's reset request (aclk)
  // down the pipeline (from stage 0)
  input  logic            r_v,
  input  logic [255:0]    r_d,
  input  logic            cc,              // a command FIFO entry freed
  input  logic            wc,              // a W FIFO entry freed
  input  logic            bp,              // a write beat answered
  input  logic            er,              // n_err (aclk)
  input  logic            hd               // the hold
);
  localparam int K    = CB / 32;
  localparam int MAXB = 16 / K;
  localparam int LB   = $clog2(MAXB);
  localparam int KW   = $clog2(K);
  localparam int WAW  = $clog2(WDD) + 1;
  localparam int RAW  = $clog2(ARD) + 1;
  localparam int CAC  = $clog2(ACQ + 1);
  localparam int CAD  = $clog2(AWD + 1);
  localparam int CW   = 16;
  localparam int NL   = RHB ? K / 2 : K;   // u_rd's FIFOs (RHB: lanes of two AXI beats a native beat)
  localparam int RLD  = RHB ? 2 * ARD : ARD;  // their entries
  localparam int RLW  = $clog2(RLD) + 1;
  localparam int KP   = WHB ? K / 2 : K;   // AXI beats a u_wd entry
  localparam int KPW  = $clog2(KP);
  localparam int WE   = KP * 288 + WHB;    // u_wd's entry: {mask, data} of KP AXI beats (WHB: and
                                           //   its half, 1 the high one, on top)
  initial begin
    if (K * 32 != CB || (K != 2 && K != 4)) $fatal(1, "otpu_hbm_cs: CB must be 64 or 128");
    if ((RHB || WHB) && K != 4) $fatal(1, "otpu_hbm_cs: RHB / WHB need CB 128");
    if (WHB && !WPOST) $fatal(1, "otpu_hbm_cs: WHB needs WPOST");
    if (ARD < 2 * MAXB || ARD != (1 << $clog2(ARD))) $fatal(1, "otpu_hbm_cs: ARD");
    if (PST < 1) $fatal(1, "otpu_hbm_cs: PST >= 1");
  end
  function automatic logic [CW-1:0] b2g(input logic [CW-1:0] b);
    return b ^ (b >> 1);
  endfunction
  function automatic logic [CW-1:0] g2b(input logic [CW-1:0] g);
    logic [CW-1:0] b;
    b[CW-1] = g[CW-1];
    for (int i = CW - 2; i >= 0; i--) b[i] = b[i + 1] ^ g[i];
    return b;
  endfunction

  // ================================================================ resets
  // the hold (the AXI half's, down the pipeline) resets the aclk side, from a register of the
  // core half's (the pipeline's DONT_TOUCH stages have no replicas: probe build 8c's hold, ~340
  // loads off stage 0, 0.82 ns of its route into u_wd's read side); the core's reset request
  // (a_req, clk) into aclk and up the pipeline (rq); the hold into clk (a_hs)
  logic a_hold = 1'b1;
  always_ff @(posedge aclk) a_hold <= hd;
  // CPY: the hold's FIFO resets from copies of a_hold, each loaded from hd in the same cycle
  // (DONT_TOUCH, so synthesis keeps them apart): one for u_cq's and u_wd's read sides (a_hx), one
  // for each u_rd's write side (a_hr[q]), a_hold for the rest (bcu1525-s2's s2syn: a_hold's 232
  // loads over 120-180 rows of SLR0, its u_rd resets -0.238 ns at 300 MHz)
  logic          a_hx;
  logic [NL-1:0] a_hr;
  if (CPY) begin : g_hcp
    (* dont_touch = "true" *) logic          a_hx_q = 1'b1;
    (* dont_touch = "true" *) logic [NL-1:0] a_hr_q = '1;
    always_ff @(posedge aclk) begin a_hx_q <= hd; a_hr_q <= {NL{hd}}; end
    assign a_hx = a_hx_q;
    assign a_hr = a_hr_q;
  end else begin : g_h1
    assign a_hx = a_hold;
    assign a_hr = {NL{a_hold}};
  end
  (* ASYNC_REG = "TRUE" *) logic a_rs1 = 1'b0, a_rs2 = 1'b0;
  (* ASYNC_REG = "TRUE" *) logic a_hs1 = 1'b1, a_hs2 = 1'b1;
  logic a_req = 1'b0, a_crst = 1'b1;
  always_ff @(posedge aclk) begin a_rs1 <= a_req; a_rs2 <= a_rs1; end
  assign rq = a_rs2;
  always_ff @(posedge clk) begin
    a_req <= rst || (a_req && !a_hs2);
    a_hs1 <= a_hold; a_hs2 <= a_hs1;
    a_crst <= rst || a_req || a_hs2;
  end
  assign o_crst = a_crst;

  // ================================================================ clock crossings
  logic                 cq_wv, cq_wr, cq_rv, cq_rr, c_sq;
  logic [CAW+XSW+3:0]   cq_rd;
  logic                 wd_wv, wd_wr, wd_rv, wd_rr;
  logic [WE-1:0]        wd_wd, wd_rd;
  logic [WAW-1:0]       wd_wused;
  logic [NL-1:0]        rd_wv, rd_wr, rd_rv;
  logic [NL-1:0][255:0] rd_rd;
  logic [RLW-1:0]       rd_wused;
  logic                 rd_pop;
  // (the crossing FIFOs whole, for otpu_hbm_ch.tcl's cells by name, as the halves above)
  (* keep_hierarchy = "yes" *)
  otpu_afifo #(.W(CAW + XSW + 4), .DEPTH(CQD)) u_cq (.wclk(clk), .wrst(a_hs2), .wvalid(cq_wv),
    .wready(cq_wr), .wdata({c_sq, n_cwe, n_csrc, n_cslot, n_caddr}), .wslot('0), .wcommit(), .wused(),
    .rclk(aclk), .rrst(a_hx), .rvalid(cq_rv), .rready(cq_rr), .rdata(cq_rd), .rused());
  (* keep_hierarchy = "yes" *)
  otpu_afifo #(.W(WE), .DEPTH(WDD)) u_wd (.wclk(clk), .wrst(a_hs2), .wvalid(wd_wv),
    .wready(wd_wr), .wdata(wd_wd), .wslot('0), .wcommit(), .wused(wd_wused), .rclk(aclk),
    .rrst(a_hx), .rvalid(wd_rv), .rready(wd_rr), .rdata(wd_rd), .rused());
  logic wd_h = 1'b0;                       // WHB: the next half u_wd takes is a beat's high one
  always_ff @(posedge clk) wd_h <= (a_crst || !WHB) ? 1'b0 : wd_h ^ wd_wv;
  if (WHB) begin : g_wdh
    assign wd_wd = {wd_h, n_wmask[CB/2-1:0], n_wdata[CB*4-1:0]};
  end else begin : g_wdf
    assign wd_wd = {n_wmask, n_wdata};
  end
  // u_rd: one FIFO an AXI beat of the native beat (RHB: a lane, AXI beats q and q + K / 2; RWR: a
  // registered wready and write address, so the RAMs' write enables start from flip-flops at 450
  // MHz; the credits keep them from filling, and nothing is written in a hold); popped together at
  // the core clock. RHB: a beat's low half starts only with the whole beat in u_rd (each lane's
  // read side sees both its entries) and neither the core's reset nor the hold up (rst, a_crst,
  // a_hs2), and its high half goes the next cycle whatever the resets (rh_p): the read pointers
  // reset (a_hs2) a cycle after a low half at the earliest, so the halves never straddle a reset
  // and the adapter's toggle stays in step across one of 2 cycles or more
  logic [NL-1:0][RLW-1:0] rd_ru;
  logic           rh_p = 1'b0, rd_lo;
  if (RHB) begin : g_rlo
    assign rd_lo  = !rh_p && !rst && !a_crst && !a_hs2 && rd_ru[0] >= RLW'(2) && rd_ru[1] >= RLW'(2);
    assign rd_pop = rd_lo || rh_p;
  end else begin : g_rwh
    assign rd_lo  = 1'b0;
    assign rd_pop = &rd_rv;
  end
  always_ff @(posedge clk) rh_p <= rd_lo;
  for (genvar q = 0; q < NL; q++) begin : g_rd
    logic [RLW-1:0] wu;
    (* keep_hierarchy = "yes" *)
    otpu_afifo #(.W(256), .DEPTH(RLD), .RWR(1'b1), .WCP(CPY ? 4 : 1)) u_rd (.wclk(aclk), .wrst(a_hr[q]),
      .wvalid(rd_wv[q]), .wready(rd_wr[q]), .wdata(r_d), .wslot('0), .wcommit(), .wused(wu), .rclk(clk),
      .rrst(a_hs2), .rvalid(rd_rv[q]), .rready(rd_pop), .rdata(rd_rd[q]), .rused(rd_ru[q]));
    if (q == NL - 1) begin : g_wu
      assign rd_wused = wu;
    end
  end

  // ================================================================ core side (clk)
  assign n_cready = !a_crst && cq_wr;
  assign cq_wv    = n_cvalid && n_cready;
  assign n_wready = !a_crst && wd_wr;
  assign n_wfree  = a_crst ? '0 : (WAW'(WDD) - wd_wused) >> WHB;   // (WHB: entries are halves)
  assign wd_wv    = n_wvalid && n_wready;
  // a command continues the one before it: the same kind and source, the next beat, not a block's
  // first (after a reset the gather is empty, so the first command starts a burst whatever c_seq
  // says)
  logic           c_wel = 1'b0;
  logic [1:0]     c_src = '0;
  logic [CAW-1:0] c_e = '0;                // the beat after the last command's
  assign c_sq = n_cwe == c_wel && n_csrc == c_src && n_caddr == c_e && n_caddr[LB-1:0] != '0;
  always_ff @(posedge clk) if (cq_wv) begin c_wel <= n_cwe; c_src <= n_csrc; c_e <= n_caddr + 1'b1; end
  always_ff @(posedge clk) begin
    n_rvalid <= RHB ? rd_pop : rd_pop && !a_crst;   // (RHB: rd_lo's own test)
    n_rdata  <= (CB*8)'(rd_rd);              // (RHB: the low half)
  end
  // n_wdone: the answered beats' count (aclk, gray), two flip-flops, decoded
  logic [CW-1:0] wacc, wacc_g;
  (* ASYNC_REG = "TRUE" *) logic [CW-1:0] wacc_s1, wacc_s2;
`ifndef SYNTHESIS
  // +cdc_skew=P (tb_hbmch): in P percent of the first stage's samples each bit is the count's bit
  // a source cycle earlier or now, at random (the bus skew the constraints allow; otpu_mem_ch)
  int cdc_skew = 0;
  initial void'($value$plusargs("cdc_skew=%d", cdc_skew));
  logic [CW-1:0] wacc_gq;
  always_ff @(posedge aclk) wacc_gq <= wacc_g;
  function automatic logic [CW-1:0] cdc_smp(input logic [CW-1:0] now, input logic [CW-1:0] was,
                                            input logic [CW-1:0] seen);
    logic [CW-1:0] m;
    m = (cdc_skew != 0 && int'($urandom % 100) < cdc_skew) ? CW'($urandom) & (seen ^ now) : '0;
    return (now & ~m) | (was & m);
  endfunction
`endif
  always_ff @(posedge clk)
    if (a_crst) begin wacc_s1 <= '0; wacc_s2 <= '0; n_wdone <= '0; end
    else begin
`ifdef SYNTHESIS
      wacc_s1 <= wacc_g;
`else
      wacc_s1 <= cdc_smp(wacc_g, wacc_gq, wacc_s1);
`endif
      wacc_s2 <= wacc_s1;
      n_wdone <= g2b(wacc_s2);
    end
  always_ff @(posedge aclk) begin
    if (a_hold) wacc <= '0;
    else wacc <= wacc + CW'(bp);
    wacc_g <= b2g(wacc);
  end
  // n_err: the AXI half's, two flip-flops
  (* ASYNC_REG = "TRUE" *) logic e_s1 = 1'b0, e_s2 = 1'b0;
  always_ff @(posedge clk) begin
    e_s1 <= er; e_s2 <= e_s1;
  end
  assign n_err = e_s2;

  // ================================================================ aclk side
  // the command pump: u_cq's head into a register (h_*), then up the pipeline while the AXI half's
  // command FIFO has room (a credit an entry, back at its pops: cq_cr, cq_ok from a register) and
  // the record FIFO has room (for a read's record; a write waits for it too, which keeps the
  // decision off the head's bits); none while the reset request is up
  logic [CAC-1:0]  cq_cr = CAC'(ACQ), cq_n;
  logic            cq_ok = 1'b0, h_v = 1'b0, h_go, rc_wr, rc_rv, rc_pop;
  logic [CAW+XSW+3:0] h_d;                 // {seq, we, src, slot, beat}
  logic [XSW:0]    rc_rd;
  assign h_go  = h_v && cq_ok && rc_wr && !a_rs2;
  assign cq_rr = !h_v || h_go;
  assign cq_n  = cq_cr - CAC'(h_go) + CAC'(cc);
  always_ff @(posedge aclk)
    if (a_hold) begin cq_cr <= CAC'(ACQ); cq_ok <= 1'b0; h_v <= 1'b0; end
    else begin
      cq_cr <= cq_n; cq_ok <= cq_n != '0;
      if (cq_rr) begin h_v <= cq_rv; h_d <= cq_rd; end
    end
  assign c_v    = h_go;
  assign c_seq  = h_d[CAW + XSW + 3];
  assign c_we   = h_d[CAW + XSW + 2];
  assign c_post = h_d[CAW + XSW +: 2] == 2'd3;
  assign c_a    = h_d[CAW-1:0];
  // the record FIFO: each read in flight's {posted, slot}, in command order
  otpu_sfifo #(.W(XSW + 1), .DEPTH(RCD), .RO(1'b1)) u_rc (.clk(aclk), .rst(a_hold),
    .wvalid(h_go && !c_we), .wready(rc_wr), .wdata({c_post, h_d[CAW +: XSW]}), .rvalid(rc_rv),
    .rready(rc_pop), .rdata(rc_rd));

  // the W pump: u_wd's head (wp_d, wp_m) as its KP AXI beats, each a credit of the W FIFO's (w_cr,
  // w_ok from a register), at WH in two halves (the high half the cycle after the low, on the same
  // credit); none starts while the reset request is up (the bursts issued have their beats in the
  // W FIFO already). wp_d / wp_m load whenever an entry ends or none is held (wp_nb), the
  // head popped or not (wp_v says which): their ~1,150 enables (WHB ~580) from the pump's own registers, not
  // through u_wd's read side and the hold (probe build 8c's aclk: wp_q -> wp_ld -> wp_d's CE,
  // -0.244 ns after phys_opt); with u_wd empty they take the slot its writer fills, unread
  logic [CAD-1:0]  w_cr = CAD'(AWD), w_n;
  logic            w_ok = 1'b0;
  logic            wp_v = 1'b0, wp_h = 1'b0, wp_take, wp_go, wp_end, wp_ld;
  (* max_fanout = 128 *) logic wp_nb;
  logic [KPW-1:0]  wp_q = '0;
  logic [KP*256-1:0] wp_d;
  logic [KP*32-1:0]  wp_m;
  logic [255:0]    wp_bd;
  logic [31:0]     wp_bm;
  assign wp_bd   = wp_d[wp_q * 256 +: 256];
  assign wp_bm   = wp_m[wp_q * 32 +: 32];
  assign wp_take = wp_v && !wp_h && w_ok;            // an AXI beat's first part goes
  assign wp_go   = wp_take || (WH && wp_h);
  assign wp_end  = wp_go && wp_q == KPW'(KP - 1) && (!WH || wp_h);
  assign wp_nb   = !wp_v || wp_end;                  // the entry ends (or none)
  assign wp_ld   = wp_nb && wd_rv && !a_rs2 && !a_hold;
  assign wd_rr   = wp_ld;
  assign w_n     = w_cr - CAD'(wp_take) + CAD'(wc);
  always_ff @(posedge aclk) begin
    if (a_hold) begin
      w_cr <= CAD'(AWD); w_ok <= 1'b0; wp_v <= 1'b0; wp_q <= '0; wp_h <= 1'b0;
    end else begin
      w_cr <= w_n; w_ok <= w_n != '0;
      if (wp_nb) begin wp_v <= wp_ld; wp_q <= '0; wp_h <= 1'b0; end
      else if (wp_go) begin
        if (WH && !wp_h) wp_h <= 1'b1;
        else begin wp_h <= 1'b0; wp_q <= wp_q + 1'b1; end
      end
    end
    if (wp_nb) {wp_m, wp_d} <= wd_rd[KP*288-1:0];
  end
  assign w_v = wp_go;
  if (WH) begin : g_wh
    assign w_d = wp_h ? {wp_bd[255:128], wp_bm[31:16], 1'b1} : {wp_bd[127:0], wp_bm[15:0], 1'b0};
  end else begin : g_wf
    assign w_d = {wp_bd, wp_bm, 1'b0};
  end

  // the R sink: an AXI beat a cycle from the pipeline; the record FIFO's head says where its
  // native beat goes: AXI beat s_q into its u_rd FIFO, or posted; nothing written while the
  // reset request is up (the beats of reads issued before it, until the hold)
  logic [KW-1:0] s_q = '0;
  logic          s_post;
  logic [XSW-1:0] s_slot;
  assign {s_post, s_slot} = rc_rd;
  assign rc_pop = r_v && s_q == KW'(K - 1);
  always_ff @(posedge aclk)
    if (a_hold) s_q <= '0;
    else if (r_v) s_q <= s_q + 1'b1;
  for (genvar q = 0; q < NL; q++) begin : g_rdv
    assign rd_wv[q] = r_v && !s_post && (s_q & KW'(NL - 1)) == KW'(q) && !a_rs2;
  end
  // posted: PST register stages to the lane RAMs, the last one's quarter and entry in copies of
  // equal registers (DONT_TOUCH: one a lane RAM group, its fan-out the group's block RAMs; the
  // quarter one-hot with the write); p_cnt steps on the last stage's last quarter, its gray code
  // a cycle later from a register (after the quarter's write), never reset
  logic [PST-1:0]           ps_v = '0;
  logic [PST-1:0][KW-1:0]   ps_q;                  // (stages 0 .. PST - 2: the last's are pl_*)
  logic [PST-1:0][XSW-1:0]  ps_s;
  logic [PST-1:0][255:0]    ps_d;
  (* dont_touch = "true" *) logic [1:0][K-1:0]   pl_q = '0;
  (* dont_touch = "true" *) logic [4:0][XSW-1:0] pl_s;
  logic                     pi_v;                  // the last stage's inputs
  logic [KW-1:0]            pi_q;
  logic [XSW-1:0]           pi_s;
  if (PST == 1) begin : g_pl1
    assign pi_v = r_v && s_post && !a_rs2, pi_q = s_q, pi_s = s_slot;
  end else begin : g_pln
    assign pi_v = ps_v[PST-2], pi_q = ps_q[PST-2], pi_s = ps_s[PST-2];
  end
  always_ff @(posedge aclk) begin
    ps_v[0] <= r_v && s_post && !a_rs2;
    ps_q[0] <= s_q; ps_s[0] <= s_slot; ps_d[0] <= r_d;
    for (int i = 1; i < PST; i++) begin
      ps_v[i] <= ps_v[i - 1]; ps_q[i] <= ps_q[i - 1]; ps_s[i] <= ps_s[i - 1]; ps_d[i] <= ps_d[i - 1];
    end
    pl_q <= {2{pi_v ? K'(1) << pi_q : K'(0)}};
    pl_s <= {5{pi_s}};
  end
  assign p_we   = ps_v[PST-1];
  assign p_q    = pl_q;
  assign p_slot = pl_s;
  assign p_d    = ps_d[PST-1];
  logic [XSW+1:0] p_cnt = '0, p_g = '0;
  always_ff @(posedge aclk) begin
    if (p_we && p_q[0][K-1]) p_cnt <= p_cnt + 1'b1;
    p_g <= p_cnt ^ (p_cnt >> 1);
  end
  assign p_cntg = p_g;

  // u_rd's frees (the core clock's pops, as the last FIFO's write side sees them: what it wrote less
  // wused; RHB: its entries over 2, the native beats), a pulse each up the pipeline (rf) for the AXI
  // half's read credits
  logic [RLW-1:0] rd_wn = '0, rd_fl;
  logic [RAW-1:0] rd_fs = '0, fr_p = '0, rd_f, fr_n;
  logic           fr_v = 1'b0;
  assign rd_fl = rd_wn - rd_wused;
  assign rd_f = RAW'(rd_fl >> RHB);
  assign fr_n = fr_p + (rd_f - rd_fs) - RAW'(fr_v);
  always_ff @(posedge aclk)
    if (a_hold) begin rd_wn <= '0; rd_fs <= '0; fr_p <= '0; fr_v <= 1'b0; end
    else begin
      rd_wn <= rd_wn + RLW'(rd_wv[NL-1]);
      rd_fs <= rd_f;
      fr_p  <= fr_n;
      fr_v  <= fr_n != '0;
    end
  assign rf = fr_v;

`ifndef SYNTHESIS
  always_ff @(posedge aclk) if (!a_hold) begin
    for (int q = 0; q < NL; q++)
      if (rd_wv[q] && !rd_wr[q]) $error("otpu_hbm_ch %0d: read data with u_rd full (credits)", PC);
    if (r_v && !rc_rv) $error("otpu_hbm_ch %0d: R without a read in the record FIFO", PC);
  end
  // WHB: u_wd's entries alternate low and high halves (the pump's own count of them, wp_x)
  logic wp_x = 1'b0;
  always_ff @(posedge aclk)
    if (a_hold || !WHB) wp_x <= 1'b0;
    else if (wp_ld) begin
      if (wd_rd[WE-1] != wp_x)
        $error("otpu_hbm_ch %0d: a W entry's %s half where its %s was due", PC,
               wd_rd[WE-1] ? "high" : "low", wp_x ? "high" : "low");
      wp_x <= !wp_x;
    end
  // RHB: a beat's high half in u_rd when it goes (rd_lo saw both lanes' entries)
  always_ff @(posedge clk)
    if (RHB && rh_p && !(&rd_rv) && !a_hs1 && !a_hs2)
      $error("otpu_hbm_ch %0d: a read beat's high half not in u_rd", PC);
  always_ff @(posedge clk)
    if (WPOST && n_wvalid && !a_hs2 && !a_crst && !wd_wr)
      $error("otpu_hbm_ch %0d: a posted write into a full u_wd (n_wfree)", PC);
`endif
endmodule

// The bridge's AXI half (otpu_hbm_ch): the command FIFO (u_cf, the core half's commands on its
// credits), the gather, the issue register and its test, the hazard counts, the bursts in flight
// (rbq, wbq, wbn), u_rd's credits (rcr), the W FIFO (u_wf, the halves joined) and the W engine, B
// and n_wdone's pulses, the drain and the hold, the response checks.
module otpu_hbm_ac #(
  parameter int CB  = 128,
  parameter int CAW = 22,
  parameter logic [4:0] PC = 5'd0,
  parameter int OAW = 34,
  parameter int ARD = 64,
  parameter int WOT = 16,
  parameter int NBK = 32,
  parameter int N   = 0,
  parameter bit WH  = 1'b0,
  parameter int ACQ = 16,
  parameter int AWD = 32,
  parameter int BHW = 0,
  parameter bit CPY = 1'b0                 // the W FIFO's head address in copies (otpu_sfifo HCP),
                                           // the W join's enables from w_c
) (
  input  logic            aclk,
  input  logic            arst,
  input  logic [5:0]      mwait,
  // up the pipeline (from stage N)
  input  logic            c_v,
  input  logic            c_we,
  input  logic            c_seq,
  input  logic            c_post,
  input  logic [CAW-1:0]  c_a,
  input  logic            w_v,
  input  logic [(WH ? 145 : 289)-1:0] w_d,
  input  logic [(CPY ? (WH ? 3 : 1) : 1)-1:0][1:0] w_c,   // CPY: {w_v, w_d[0]} in copies
  input  logic            rf,
  input  logic            rq,
  // down the pipeline (into stage N)
  output logic            r_v,
  output logic [255:0]    r_d,
  output logic            cc,
  output logic            wc,
  output logic            bp,
  output logic            er,
  output logic            hd,
  // the HBM's AXI3 port
  output logic            m_awvalid,
  input  logic            m_awready,
  output logic [OAW-1:0]  m_awaddr,
  output logic [3:0]      m_awlen,
  output logic            m_wvalid,
  input  logic            m_wready,
  output logic [255:0]    m_wdata,
  output logic [31:0]     m_wstrb,
  output logic            m_wlast,
  input  logic            m_bvalid,
  output logic            m_bready,
  input  logic [1:0]      m_bresp,
  output logic            m_arvalid,
  input  logic            m_arready,
  output logic [OAW-1:0]  m_araddr,
  output logic [3:0]      m_arlen,
  input  logic            m_rvalid,
  output logic            m_rready,
  input  logic [255:0]    m_rdata,
  input  logic [1:0]      m_rresp,
  input  logic            m_rlast,
  output logic [3:0]      ev,
  output logic [$clog2(16 / (CB / 32)):0] ev_n
);
  localparam int K    = CB / 32;           // AXI beats per native beat
  localparam int MAXB = 16 / K;            // native beats per burst (16 AXI beats)
  localparam int LB   = $clog2(MAXB);      // the block's beat bits
  localparam int KW   = $clog2(K);
  localparam int AB   = $clog2(CB);        // byte bits of a native beat
  localparam int PCW  = OAW - CAW - AB;    // PC's bits in the AXI address
  localparam int HZB  = BHW != 0 ? BHW - 1 - AB : 0;     // (BHW: the beat bit from which on)
  localparam int NW   = LB + 1;            // a burst's native beats, 1 .. MAXB
  localparam int BKW  = $clog2(NBK);
  localparam int RBW  = $clog2(ARD) + 1;   // read beats out (rcr), 0 .. ARD
  localparam int WBW  = $clog2(WOT) + 1;   // write bursts in flight, 0 .. WOT
  localparam int AWN  = AWD / K;           // the W FIFO's native beats
  localparam int WAW  = $clog2(AWN) + 2;   // its native beats' counts
  localparam int DPW  = $clog2(WOT * MAXB) + 2;  // beats answered, not yet counted
  localparam int HL   = 2 * (N + 1) + 4 > 16 ? 2 * (N + 1) + 4 : 16;  // the hold's cycles
  localparam int HW   = $clog2(HL + 1);
  initial begin
    if (K * 32 != CB || (K != 2 && K != 4)) $fatal(1, "otpu_hbm_ac: CB must be 64 or 128");
    if (PCW < 0 || PCW < 5 && (PC >> PCW) != 5'd0)
      $fatal(1, "otpu_hbm_ac: CAW + log2(CB) beyond OAW, or PC beyond its OAW - CAW - log2(CB) bits");
    if (ARD < 2 * MAXB || ARD != (1 << $clog2(ARD))) $fatal(1, "otpu_hbm_ac: ARD");
    if (NBK != (1 << BKW) || BKW > CAW - LB) $fatal(1, "otpu_hbm_ac: NBK");
    if (BHW != 0 && (HZB < LB || HZB >= CAW)) $fatal(1, "otpu_hbm_ac: BHW (a block's bit .. the channel's)");
    if (AWN < MAXB || AWD != (1 << $clog2(AWD))) $fatal(1, "otpu_hbm_ac: AWD (a burst's beats at least)");
  end
  // the hazard bucket of a beat: its 512-byte block's index, XOR-folded
  function automatic logic [BKW-1:0] bkh(input logic [CAW-1:0] a);
    logic [BKW-1:0] h;
    h = '0;
    for (int i = LB; i < CAW; i += BKW)
      for (int j = 0; j < BKW; j++)
        if (i + j < CAW) h[j] ^= a[i + j];
    return h;
  endfunction

  // a read the hazards leave alone (BHW): from the window's upper half up
  function automatic logic nhz(input logic [CAW-1:0] a);
    return BHW != 0 && (a >> HZB) != '0;
  endfunction

  // ================================================================ resets
  // arst registered once (keep); the core's reset request (rq, up the pipeline). a_stop: nothing
  // more is issued (the drain, then the hold); a_hold: the FIFOs' reset, down the pipeline (hd).
  // (urq_h: a copy for the hazard counts' ~450 flip-flops; out of context the one register's
  // fanout was aclk's worst path)
  (* keep = "true" *) logic urq = 1'b1, urq_h = 1'b1;
  always_ff @(posedge aclk) begin urq <= arst; urq_h <= arst; end
  logic a_stop = 1'b1, a_hold = 1'b1;
  logic [HW-1:0] a_hcnt = HW'(HL);
  logic a_dn;                              // nothing of the core's in flight (see the drain)
  always_ff @(posedge aclk) begin
    if (urq) begin
      a_stop <= 1'b1; a_hold <= 1'b1; a_hcnt <= HW'(HL);
    end else if (!a_stop) begin
      a_stop <= rq;
    end else if (!a_hold) begin
      if (a_dn) begin a_hold <= 1'b1; a_hcnt <= HW'(HL); end
    end else begin
      a_hcnt <= rq ? HW'(HL) : (a_hcnt != 0) ? a_hcnt - 1'b1 : a_hcnt;
      if (a_hcnt == 0 && !rq) begin a_hold <= 1'b0; a_stop <= 1'b0; end
    end
  end
  assign hd = a_hold;

  // ================================================================ the command FIFO
  // the core half's commands, on its credits (an entry each, back at the pop: cc); its head from a
  // register (h_*)
  logic           h_v, h_seq, h_we, h_post, h_pop, cf_wr;
  logic [CAW-1:0] h_a;
  otpu_sfifo #(.W(CAW + 3), .DEPTH(ACQ), .RO(1'b1)) u_cf (.clk(aclk), .rst(a_hold), .wvalid(c_v),
    .wready(cf_wr), .wdata({c_seq, c_we, c_post, c_a}), .rvalid(h_v), .rready(h_pop),
    .rdata({h_seq, h_we, h_post, h_a}));
  always_ff @(posedge aclk) cc <= h_pop && !a_hold;

  // the gather: the burst being collected (start g_a, g_n beats, bucket g_bk; g_el: the block
  // offset of the beat after its last, 0 once the block is complete). The head joins it when it
  // continues the command before it, which is the gather's last beat whenever the gather holds one
  logic           g_v, g_we, g_post, g_nh, g_mrg, g_end, g_mv;
  logic [CAW-1:0] g_a;
  logic [LB-1:0]  g_el;
  logic [NW-1:0]  g_n;
  logic [BKW-1:0] g_bk;
  logic [5:0]     g_age;                   // aclk cycles since the last beat joined
  logic [5:0]     mw;                      // mwait, registered (static: off the gather's path)
  logic           g_old;                   // g_age >= mw
  // the issue register (i_*) and the issue's state (below)
  logic           i_v, i_we, i_post, i_nh, i_room, i_haz, i_go, ar_go, aw_go;
  logic [CAW-1:0] i_a;
  logic [NW-1:0]  i_n;
  logic [BKW-1:0] i_bk;
  (* keep = "true" *) logic [BKW-1:0] i_bkh;  // i_bk for the hazard lookup alone (probe 7f: i_bk to
                                         // i_ok -0.128 ns, p_oh's decoder on the same nets)
  logic           ar_v, aw_v, rok, rbq_wr, wbq_wr, wbn_wr;
  logic [WAW-1:0] wav;                     // native beats in the W FIFO not yet given to a burst
  logic [NBK-1:0] rnz, wnz;                // a read / write burst to the bucket in flight
  assign g_mrg = g_v && h_v && h_seq;
  assign g_end = g_el == '0 || (h_v && !h_seq) || (!h_v && g_old);
  assign g_mv  = !a_stop && g_v && g_end && !i_v;
  assign h_pop = !a_stop && h_v && (!g_v || g_mrg || g_mv);
  always_ff @(posedge aclk) begin
    mw <= mwait;
    if (a_stop) g_v <= 1'b0;
    else if (h_pop && !g_mrg) begin                     // a new burst from the head
      g_v <= 1'b1; g_we <= h_we; g_post <= h_post; g_nh <= !h_we && nhz(h_a);
      g_a <= h_a; g_el <= h_a[LB-1:0] + 1'b1;
      g_n <= NW'(1); g_bk <= bkh(h_a); g_age <= '0; g_old <= mw == '0;
    end else if (g_mrg) begin
      g_el <= g_el + 1'b1; g_n <= g_n + 1'b1; g_age <= '0; g_old <= mw == '0;
    end else begin
      if (g_mv) g_v <= 1'b0;
      if (g_age != '1) begin g_age <= g_age + 1'b1; g_old <= g_age + 1'b1 >= mw; end
    end
  end

  // the issue register: a complete burst, out when it has room and no hazard holds it. The test
  // is registered (i_ok, a cycle old): while the register waits nothing else issues, so its room
  // (credits, W beats, FIFO entries, the AR / AW registers) only grows and its hazards only clear;
  // a burst just loaded is tested in the cycle after its load, when the last issue's effects are
  // in (i_ok is stale only in the cycle after an issue, when the register is empty). A posted read
  // takes no u_rd credits
  logic i_ok = 1'b0;
  assign i_room = i_we ? (!aw_v && wav >= WAW'(i_n) && wbq_wr && wbn_wr) : (!ar_v && (i_post || rok) && rbq_wr);
  assign i_haz  = i_we ? rnz[i_bkh] : wnz[i_bkh] && !i_nh;
  assign i_go   = !a_stop && i_v && i_ok;
  always_ff @(posedge aclk) i_ok <= i_v && i_room && !i_haz;
  assign ar_go  = i_go && !i_we;
  assign aw_go  = i_go && i_we;
  always_ff @(posedge aclk) begin
    if (a_stop) i_v <= 1'b0;
    else if (g_mv) i_v <= 1'b1;
    else if (i_go) i_v <= 1'b0;
    if (g_mv) begin
      i_we <= g_we; i_post <= g_post; i_nh <= g_nh; i_a <= g_a; i_n <= g_n; i_bk <= g_bk;
      i_bkh <= g_bk;
    end
  end
  // an issue's effects, a cycle after it (p_*): the burst into rbq or wbq and wbn, the credits,
  // the W beats it claims, the hazard counts (p_oh: its bucket, one-hot)
  logic           p_ar = 1'b0, p_aw = 1'b0, p_post, p_nh;
  logic [NW-1:0]  p_n;
  logic [BKW-1:0] p_bk;
  logic [NBK-1:0] p_oh;
  always_ff @(posedge aclk) begin
    p_ar   <= ar_go && !urq;
    p_aw   <= aw_go && !urq;
    p_post <= i_post;
    p_nh   <= i_nh;
    p_n    <= i_n;
    p_bk   <= i_bk;
    p_oh   <= NBK'(1) << i_bk;
  end

  // ---------------------------------------------------------------- reads
  logic [CAW-1:0] ar_a;
  logic [NW-1:0]  ar_n;
  assign m_arvalid = ar_v;
  assign m_araddr  = OAW'({PC, ar_a, AB'(0)});
  assign m_arlen   = 4'(ar_n * K - 1);
  always_ff @(posedge aclk) begin
    if (urq) ar_v <= 1'b0;
    else if (ar_go) ar_v <= 1'b1;
    else if (m_arready) ar_v <= 1'b0;
    if (ar_go) begin ar_a <= i_a; ar_n <= i_n; end
  end

  // R: a register, then down the pipeline (the core half's sink); the bursts in flight in order
  // (rbq), each ending at its RLAST
  logic           r_l, r_e, r_bend;
  logic [3:0]     r_bc;                    // the AXI beat in the burst
  logic           rbq_rv;
  logic [BKW+NW:0] rbq_rd;                 // {no hazard (BHW), bucket, beats}
  logic [RBW-1:0] rcr;                     // sources 0-2's beats issued, not yet freed from u_rd
  logic           r_bad;
  assign m_rready = 1'b1;
  always_ff @(posedge aclk) begin
    r_v <= m_rvalid && !urq;
    r_d <= m_rdata;
    r_l <= m_rlast;
    r_e <= m_rresp != 2'b00;
  end
  assign r_bend = r_v && r_l && rbq_rv;
  always_ff @(posedge aclk)
    if (urq) r_bc <= '0;
    else if (r_v) r_bc <= r_bend ? 4'd0 : r_bc + 1'b1;
  // the check, a cycle late: a beat without a burst in flight, RLAST off its burst's last beat,
  // RRESP not OKAY
  logic r_cv = 1'b0, r_cl, r_cx, r_ce;
  always_ff @(posedge aclk) begin
    r_cv <= r_v && !urq;
    r_cl <= r_l;
    r_cx <= r_bc == 4'(rbq_rd[NW-1:0] * K - 1);
    r_ce <= r_e || !rbq_rv;
  end
  assign r_bad = r_cv && (r_ce || r_cl != r_cx);
  otpu_sfifo #(.W(BKW + NW + 1), .DEPTH(ARD), .RO(1'b1)) u_rbq (.clk(aclk), .rst(urq),
    .wvalid(p_ar), .wready(rbq_wr), .wdata({p_nh, p_bk, p_n}), .rvalid(rbq_rv), .rready(r_bend),
    .rdata(rbq_rd));
  // credits: room in u_rd for MAXB beats beyond every read of sources 0-2 issued and not yet freed
  // (in flight, in the pipeline both ways, in u_rd: rcr, exact), the test on the count after this
  // cycle's issue and free (at 400 ns of latency the free a cycle later cost 0.1 GB/s a channel)
  logic [RBW-1:0] rcr_n;
  assign rcr_n = rcr + ((p_ar && !p_post) ? RBW'(p_n) : '0) - RBW'(rf);
  always_ff @(posedge aclk)
    if (a_hold) begin rcr <= '0; rok <= 1'b0; end
    else begin
      rcr <= rcr_n;
      rok <= rcr_n <= RBW'(ARD - MAXB);
    end

  // ---------------------------------------------------------------- writes
  logic [CAW-1:0] aw_a;
  logic [NW-1:0]  aw_n;
  assign m_awvalid = aw_v;
  assign m_awaddr  = OAW'({PC, aw_a, AB'(0)});
  assign m_awlen   = 4'(aw_n * K - 1);
  always_ff @(posedge aclk) begin
    if (urq) aw_v <= 1'b0;
    else if (aw_go) aw_v <= 1'b1;
    else if (m_awready) aw_v <= 1'b0;
    if (aw_go) begin aw_a <= i_a; aw_n <= i_n; end
  end
  // the W FIFO (u_wf): AXI beats {strobes, data}, the halves joined at WH (the low half held in
  // w_lo); an entry popped is a credit back (wc). wd_rused: the native beats all in it, less those
  // the W engine has started on
  logic           wf_wv, wf_wr, wf_rv, wf_rr;
  logic [287:0]   wf_wd, wf_rd;
  logic [KW-1:0]  wf_q = '0;               // the AXI beat of the native beat it takes next
  logic [WAW-1:0] wf_in = '0, wf_ld = '0, wd_rused;
  // CPY: the enables from w_c, stage N's copies of w_v and w_d[0] (at WH copy g for w_lo's half g,
  // copy 2 for the W FIFO's write; B1, bcu1525-x2: stage N's w_v into w_lo's 144 enables and the
  // W FIFO's RAMs' write enables)
  if (WH) begin : g_wj
    logic [143:0] w_lo;
    if (CPY) begin : g_c
      always_ff @(posedge aclk)
        for (int b = 0; b < 144; b++) if (w_c[b / 72][1] && !w_c[b / 72][0]) w_lo[b] <= w_d[b + 1];
      assign wf_wv = w_c[2][1] && w_c[2][0];
    end else begin : g_1
      always_ff @(posedge aclk) if (w_v && !w_d[0]) w_lo <= w_d[144:1];
      assign wf_wv = w_v && w_d[0];
    end
    assign wf_wd = {w_d[16:1], w_lo[15:0], w_d[144:17], w_lo[143:16]};
  end else begin : g_wjf
    assign wf_wv = CPY ? w_c[0][1] : w_v;
    assign wf_wd = {w_d[32:1], w_d[288:33]};
  end
  otpu_sfifo #(.W(288), .DEPTH(AWD), .RO(1'b1), .HCP(CPY ? 4 : 1)) u_wf (.clk(aclk), .rst(a_hold),
    .wvalid(wf_wv), .wready(wf_wr), .wdata(wf_wd), .rvalid(wf_rv), .rready(wf_rr), .rdata(wf_rd));
  always_ff @(posedge aclk) wc <= wf_rr && wf_rv && !a_hold;
  assign wd_rused = wf_in - wf_ld;
  // the claims: wcl, given to issued bursts, not yet started on (wav never above the beats there)
  logic [WAW-1:0] wcl;
  logic           w_ld;
  always_ff @(posedge aclk)
    if (a_hold) begin wcl <= '0; wav <= '0; wf_q <= '0; wf_in <= '0; wf_ld <= '0; end
    else begin
      if (wf_wv) wf_q <= wf_q + 1'b1;
      wf_in <= wf_in + WAW'(wf_wv && wf_q == KW'(K - 1));
      wf_ld <= wf_ld + WAW'(w_ld);
      wcl <= wcl + (p_aw ? WAW'(p_n) : '0) - WAW'(w_ld);
      wav <= wd_rused - wcl - (p_aw ? WAW'(p_n) : '0);
    end

  // the W engine: each issued burst's beats (wbn: their count), K AXI beats per native beat, from
  // the W FIFO's head (w_ld: a native beat started on)
  logic           wbn_rv, w_bv, w_nb;
  logic [NW-1:0]  wbn_rd, w_left;          // native beats of the burst left, this one's included
  logic [KW-1:0]  w_q;
  otpu_sfifo #(.W(NW), .DEPTH(WOT), .RO(1'b1)) u_wbn (.clk(aclk), .rst(urq), .wvalid(p_aw),
    .wready(wbn_wr), .wdata(p_n), .rvalid(wbn_rv), .rready(w_nb && !(w_bv && w_left != NW'(1)) && wbn_rv),
    .rdata(wbn_rd));
  assign m_wvalid = w_bv;
  assign m_wdata  = wf_rd[255:0];
  assign m_wstrb  = wf_rd[287:256];
  assign m_wlast  = w_q == KW'(K - 1) && w_left == NW'(1);
  assign wf_rr    = w_bv && m_wready;
  assign w_nb     = !w_bv || (m_wready && w_q == KW'(K - 1));  // the native beat ends (or none)
  assign w_ld     = w_nb && ((w_bv && w_left != NW'(1)) || wbn_rv);
  always_ff @(posedge aclk) begin
    if (urq) begin w_bv <= 1'b0; w_q <= '0; end
    else begin
      if (w_nb) begin
        w_bv <= w_ld;
        w_q  <= '0;
        if (w_ld) w_left <= (w_bv && w_left != NW'(1)) ? w_left - 1'b1 : wbn_rd;
      end else if (m_wready) w_q <= w_q + 1'b1;
    end
  end

  // B: a register; the bursts in flight in order (wbq)
  logic           b_v, b_e;
  logic           wbq_rv;
  logic [BKW+NW-1:0] wbq_rd;
  logic [DPW-1:0] wd_pend;                 // beats answered, not yet sent down (bp)
  assign m_bready = 1'b1;
  always_ff @(posedge aclk) begin
    b_v <= m_bvalid && !urq;
    b_e <= m_bresp != 2'b00;
  end
  otpu_sfifo #(.W(BKW + NW), .DEPTH(WOT), .RO(1'b1)) u_wbq (.clk(aclk), .rst(urq),
    .wvalid(p_aw), .wready(wbq_wr), .wdata({p_bk, p_n}), .rvalid(wbq_rv), .rready(b_v),
    .rdata(wbq_rd));
  // (a beat leaves wd_pend into wd_mv, a pulse down the pipeline into the core half's count)
  logic wd_mv;
  always_ff @(posedge aclk) begin
    if (a_hold) begin wd_pend <= '0; wd_mv <= 1'b0; end
    else begin
      wd_mv   <= wd_pend != 0;
      wd_pend <= wd_pend + ((b_v && wbq_rv && !a_stop) ? DPW'(wbq_rd[NW-1:0]) : '0) - DPW'(wd_pend != 0);
    end
  end
  assign bp = wd_mv;

  // ---------------------------------------------------------------- hazards
  // per bucket: read bursts in flight (from their AR to their last R beat), write bursts in
  // flight (from their AW to their B). An issue counts a cycle after it (p_*), an end a cycle
  // after it too (q_*, one-hot): the count is up when the next burst is tested, and down a cycle
  // late (a hazard held a cycle longer); a read from the window's upper half up (BHW) not at all
  logic [NBK-1:0] q_roh = '0, q_woh = '0;
  always_ff @(posedge aclk) begin
    q_roh <= (r_bend && !urq && !rbq_rd[NW + BKW]) ? NBK'(1) << rbq_rd[NW +: BKW] : '0;
    q_woh <= (b_v && wbq_rv && !urq) ? NBK'(1) << wbq_rd[NW +: BKW] : '0;
  end
  logic [NBK-1:0][RBW-1:0] rcnt;
  logic [NBK-1:0][WBW-1:0] wcnt;
  always_ff @(posedge aclk)
    for (int b = 0; b < NBK; b++) begin
      logic [RBW-1:0] rn;
      logic [WBW-1:0] wn;
      rn = rcnt[b] + RBW'(p_ar && !p_nh && p_oh[b]) - RBW'(q_roh[b]);
      wn = wcnt[b] + WBW'(p_aw && p_oh[b]) - WBW'(q_woh[b]);
      if (urq_h) begin rcnt[b] <= '0; wcnt[b] <= '0; rnz[b] <= 1'b0; wnz[b] <= 1'b0; end
      else begin rcnt[b] <= rn; wcnt[b] <= wn; rnz[b] <= rn != 0; wnz[b] <= wn != 0; end
    end

  // the drain is over: no burst in flight (a burst enters rbq / wbq a cycle after its AR / AW
  // (p_ar, p_aw), and a write burst's B comes after its last W beat, so wbq covers the W engine);
  // the R beats in the pipeline reach the core half before the hold, which follows them down
  assign a_dn = !rbq_rv && !wbq_rv && !p_ar && !p_aw;

  assign ev   = {i_v && i_we && i_haz && !a_stop, i_v && !i_we && !i_post && !rok && !a_stop, aw_go, ar_go};
  assign ev_n = i_n;

  // ---------------------------------------------------------------- errors
  logic c_err = 1'b0;
  always_ff @(posedge aclk)
    if (urq) c_err <= 1'b0;
    else if (r_bad || (b_v && (!wbq_rv || b_e))) c_err <= 1'b1;
  assign er = c_err;

`ifndef SYNTHESIS
  always_ff @(posedge aclk) if (!urq) begin
    if (c_v && !cf_wr && !a_hold) $error("otpu_hbm_ch %0d: the command FIFO overrun (credits)", PC);
    if (wf_wv && !wf_wr && !a_hold) $error("otpu_hbm_ch %0d: the W FIFO overrun (credits)", PC);
    if (w_bv && !wf_rv) $error("otpu_hbm_ch %0d: a W beat not in the W FIFO (wav)", PC);
    if (r_v && !rbq_rv) $error("otpu_hbm_ch %0d: R without a read burst in flight", PC);
    if (r_v && rbq_rv && r_l != (r_bc == 4'(rbq_rd[NW-1:0] * K - 1)))
      $error("otpu_hbm_ch %0d: RLAST off the burst's last beat", PC);
    if (b_v && !wbq_rv) $error("otpu_hbm_ch %0d: B without a write burst in flight", PC);
    if (r_v && r_e) $error("otpu_hbm_ch %0d: RRESP not OKAY", PC);
    if (b_v && b_e) $error("otpu_hbm_ch %0d: BRESP not OKAY", PC);
    if (i_go && i_n > NW'(MAXB)) $error("otpu_hbm_ch %0d: a burst of %0d beats", PC, i_n);
    if (rcr > RBW'(ARD)) $error("otpu_hbm_ch %0d: read slots overrun", PC);
  end
`endif
endmodule
