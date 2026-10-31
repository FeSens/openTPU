// One openTPU memory channel on HBM2 (VCU128): the otpu_mem_ch controller side (two LiteDRAM
// native ports, c_*) -> 2 x NS AXI3 masters to the ports of the Xilinx HBM IP (PG276). All in the
// uclk domain (250 MHz on the FK35, 225 MHz on the VCU128). Description, address function and
// limitations: README.md next to it.
//
// Native port p (= bit 7 of the beat number, chosen by otpu_mem_ch) is served by NS independent
// AXI ports ("sub-ports" k = 0..NS-1, AXI port a = p + 2k): a beat goes to a sub-port by the high bits
// of the number of its pseudo-channel (otpu_hbm_amap), so sub-port k only accesses the pseudo-channels
// of its 1/NS of the channel -- on the board, those of its AXI port's mini-switch (PCB 4, NS 4: the
// 4 pseudo-channels 4k..4k+3 of the stack, AXI port 16 j + 4 k + p of the channel j = stack j).
// The channel is 2^(22+PCB) beats of 64 bytes: BAW = 25 (PCB 3, 2 GiB; the default), 26 (PCB 4, 4 GiB).
// The stream of adjacent beats of one native port (a series of 128 beats) is split between the NS AXI ports:
// a native port accepts a beat every uclk cycle (64 bytes), an AXI port 32 bytes per cycle. Equal beats are
// always in the same sub-port, so the sub-port keeps the per-beat order, as the port did before; the read data
// of a native port are collected from the sub-ports in the order of its commands (FIFO of sub-port numbers).
// A beat (64 bytes) = a packet of two 256-bit words: AxLEN = 1, AxSIZE = 5, INCR.
//
// Command pipeline of a sub-port (250 MHz uclk, FK35: every c_* input goes into a register first and
// c_cmd_ready comes from registers, so no c_* input is more than a LUT or two from a flip-flop):
// F (the native port's input registers: the accepted command and its write data, one cycle; credits
// instead of a ready chain) -> CQ (the sub-port's command queue) -> A (the queue head's beat is compared
// with all outstanding operations of the sub-port -- CAM, the result goes into the dependency vector) ->
// H (head: issues AR or AW, when its vector is empty) -> FIFO AR/AW (outputs from registers) -> HBM. The
// HBM read and write channels are not ordered against each other, so the per-beat order is kept here:
//   - a write is outstanding until its B (write slot = AWID, released by B);
//   - a read is outstanding until its last R (read slot = ARID);
//   - any command waits for an outstanding write to the same beat (RAW, WAW), a write also waits for an
//     outstanding read of the same beat (WAR); a read after a read does not wait.
// A command is compared with the tables in the cycle it moves from CQ into A, and with the operation that
// H issues in the same cycle (it is not in the tables yet) -- by a separate comparator; a dependency bit
// clears when the operation completes. So the check at the head is a register (h_ok), without a CAM on
// the way from c_cmd to the head. (Before the pipelining, A took the command from c_cmd in the accept
// cycle: AR was on the bus 3 cycles after the accept, now 5.)
//
// Write: otpu_mem_ch puts the write data into its port FIFO in the cycle in which the command enters the
// port's command queue, so by the time a write command is on c_cmd, its data are already at the
// head of c_wdata (the data go in the order of the write commands). A write command is accepted only together
// with its data (c_wdata_ready = accept of a write command); the data go through F into the sub-port's FIFO
// WQ (one cycle after the accept, together with the command into CQ) and wait there, then go to W
// after their AW (W never overtakes AW, 2 words, WLAST on the second). c_wdata_we -> WSTRB.
//
// Read: when an AR is issued, a slot of the sub-port's ring buffer is reserved for it (ARID = slot),
// so R is always accepted (RREADY = 1), in any ID order and with interleaving of packets of different
// IDs. The data go to otpu_mem_ch in the order of the native port's commands (the head of the ring of the
// sub-port named at the head of the sub-port number FIFO, when both words have arrived), without
// backpressure; the two native ports share one bus c_rdata_data: the arbiter (round robin on a conflict)
// issues at most one beat per cycle, valid of exactly one port. c_rdata_* leave through two register
// stages (the second one only to cross to otpu_mem_ch: a pure copy).
//
// Errors: RRESP/BRESP != OKAY -- the counter st_err_cnt (saturating) and the flag st_err; an HBM protocol
// violation (R/B with an unknown ID, a wrong RLAST) -- the flag st_perr. Both are held until urst.
// The reset urst is synchronous.
module otpu_hbm_port #(
  parameter logic [32:0] CH_BASE = 33'h0, // byte address of the channel in the HBM (a multiple of the channel size; CH_BASE[32] -- the stack at PCB 4)
  parameter int  PCB = 3,                 // log2 of pseudo-channels per channel: 4 -- 16 (4 GiB, one stack), 3 -- 8 (2 GiB), 2 -- 4 (1 GiB)
  parameter int  BAW = 25,                // bits of the channel's beat number (c_cmd_addr): >= 22 + PCB (25: 2 GiB, 26: 4 GiB = DAW 33)
  parameter int  HAW = 0,                 // bits of the beat in the dependency CAM (tables, comparators): 0 -- BAW (exact), or 1..BAW (the top bits
                                          // dropped: two beats equal in the low HAW bits wait for each other -- a delay, never wrong data)
  parameter int  PC_GRAN = 0,             // log2 of adjacent beats in a row in one pseudo-channel
  parameter bit  PC_XOR  = 1'b1,          // pseudo-channel number XOR the next PCB bits
  parameter int  NS = 1,                  // AXI ports per native port: 1, 2, 4
  parameter int  RD = 32,                 // read slots per AXI port (ARID, buffer), 2^k, 4..64
  parameter int  WD = 32,                 // write slots per AXI port (AWID), 2^k, 4..64
  parameter int  WQ = 16,                 // write data FIFO per AXI port, beats (credit counted: the pipelining adds ~3 cycles of residency)
  parameter int  AQ = 4,                  // AR and AW FIFO per AXI port
  localparam int NA = 2 * NS              // AXI ports: a = p + 2k (native port p, sub-port k)
) (
  input  logic              uclk,
  input  logic              urst,
  // otpu_mem_ch controller side (LiteDRAM native ports)
  input  logic [1:0]        c_cmd_valid,
  output logic [1:0]        c_cmd_ready,
  input  logic [1:0]        c_cmd_we,
  input  logic [1:0][BAW-1:0] c_cmd_addr,
  input  logic [1:0]        c_wdata_valid,
  output logic [1:0]        c_wdata_ready,
  input  logic [1:0][511:0] c_wdata_data,
  input  logic [1:0][63:0]  c_wdata_we,
  output logic [1:0]        c_rdata_valid,
  output logic [1:0][511:0] c_rdata_data,  // one bus: [1] = [0]
  // NA AXI3 masters to the HBM IP ports
  output logic [NA-1:0]        m_awvalid,
  input  logic [NA-1:0]        m_awready,
  output logic [NA-1:0][32:0]  m_awaddr,
  output logic [NA-1:0][5:0]   m_awid,
  output logic [NA-1:0][3:0]   m_awlen,
  output logic [NA-1:0][2:0]   m_awsize,
  output logic [NA-1:0][1:0]   m_awburst,
  output logic [NA-1:0]        m_wvalid,
  input  logic [NA-1:0]        m_wready,
  output logic [NA-1:0][255:0] m_wdata,
  output logic [NA-1:0][31:0]  m_wstrb,
  output logic [NA-1:0]        m_wlast,
  output logic [NA-1:0][5:0]   m_wid,           // AXI3 WID (= AWID); the HBM IP has no such pin
  output logic [NA-1:0][31:0]  m_wdata_parity,  // XOR of the bits of a byte (if parity is enabled in the IP)
  input  logic [NA-1:0]        m_bvalid,
  output logic [NA-1:0]        m_bready,
  input  logic [NA-1:0][5:0]   m_bid,
  input  logic [NA-1:0][1:0]   m_bresp,
  output logic [NA-1:0]        m_arvalid,
  input  logic [NA-1:0]        m_arready,
  output logic [NA-1:0][32:0]  m_araddr,
  output logic [NA-1:0][5:0]   m_arid,
  output logic [NA-1:0][3:0]   m_arlen,
  output logic [NA-1:0][2:0]   m_arsize,
  output logic [NA-1:0][1:0]   m_arburst,
  input  logic [NA-1:0]        m_rvalid,
  output logic [NA-1:0]        m_rready,
  input  logic [NA-1:0][5:0]   m_rid,
  input  logic [NA-1:0][255:0] m_rdata,
  input  logic [NA-1:0][1:0]   m_rresp,
  input  logic [NA-1:0]        m_rlast,
  // status (platform register), per AXI port
  output logic [NA-1:0][31:0]  st_rcmd,         // read commands accepted from otpu_mem_ch
  output logic [NA-1:0][31:0]  st_wcmd,         // write commands
  output logic [NA-1:0][6:0]   st_rout,         // reads in the HBM (AR issued, last R not arrived)
  output logic [NA-1:0][6:0]   st_wout,         // writes in the HBM (AW issued, B not arrived)
  output logic [NA-1:0][31:0]  st_dep,          // cycles in which the head waited for a beat dependency
  output logic [NA-1:0]        st_busy,         // the port holds a command or an operation
  output logic [15:0]          st_err_cnt,      // responses RRESP/BRESP != OKAY (saturating)
  output logic                 st_err,          // there was a response != OKAY
  output logic                 st_perr          // the HBM violated the protocol (ID, RLAST)
);
  localparam int RSW = $clog2(RD);
  localparam int WSW = $clog2(WD);
  localparam int KW  = (NS > 1) ? $clog2(NS) : 1;   // sub-port number (at NS = 1 -- always 0)
  localparam int TQ  = 4 * NS * RD;         // FIFO of sub-port numbers of the native port's reads
  localparam int CQ  = 8;                   // command queue per AXI port (credit counted, see F)
  localparam int QW  = 33 + 6;              // AR/AW: {address, ID}
  localparam int DW  = 64 + 512;            // write data: {WSTRB, WDATA}
  localparam int G   = PC_GRAN;
  localparam int HW  = (HAW == 0) ? BAW : HAW;   // beat bits the dependency CAM compares (tables r_beat / w_beat, eq_c, eq_a, c_rm, c_wm)
  // wide structures split into slices with their own address / enable copies (kept registers: a
  // 512-bit RAM driven by one address register was 600 loads over 3.9 ns of route at 250 MHz)
  localparam int NG  = 4;                   // slices of WQ (144 bits) and of the read-data RAMs (64 bits)
  localparam int NCP = 2;                   // copies of the CQ head per CAM table
  localparam int NHB = 4;                   // copies of h_beat per CAM table
  initial begin
    if (RD != (1 << RSW) || RD < 4 || RD > 64) $fatal(1, "otpu_hbm_port: RD = 2^k, 4..64");
    if (WD != (1 << WSW) || WD < 4 || WD > 64) $fatal(1, "otpu_hbm_port: WD = 2^k, 4..64");
    if (NS != 1 && NS != 2 && NS != 4) $fatal(1, "otpu_hbm_port: NS = 1, 2, 4");
    if (PCB < 2 || PCB > 4) $fatal(1, "otpu_hbm_port: PCB = 2, 3, 4");
    if (BAW < 22 + PCB) $fatal(1, "otpu_hbm_port: BAW < 22 + PCB (the channel is 2^(22+PCB) beats)");
    if (HW < 1 || HW > BAW) $fatal(1, "otpu_hbm_port: HAW = 0 or 1..BAW");
    if (NS > (1 << PCB)) $fatal(1, "otpu_hbm_port: NS is greater than the number of pseudo-channels");
    // the reads of a native port accepted and not yet delivered (see g_n): F, then per sub-port CQ, A, H and the ring
    if (NS > 1 && TQ < 1 + NS * (CQ + 2 + RD)) $fatal(1, "otpu_hbm_port: the FIFO of sub-port numbers is too small");
    // PCB 4: a multiple of 4 GiB, 0 or 33'h1_0000_0000 (CH_BASE[32] is the HBM stack)
    if ((CH_BASE & ((33'd1 << (28 + PCB)) - 33'd1)) != '0)
      $fatal(1, "otpu_hbm_port: CH_BASE is not a multiple of the channel size");
    if (PC_GRAN < 0 || PC_GRAN + (PC_XOR ? 2 * PCB : PCB) > 22 + PCB)
      $fatal(1, "otpu_hbm_port: PC_GRAN");
  end

  // sub-port of a beat: the high bits of the pseudo-channel number in the channel (as in otpu_hbm_amap).
  // (b >> n) cut to PCB bits instead of b[n +: PCB]: no select beyond the beat's top bit with a granule
  // so large that the XOR partner (not used with PC_XOR = 0) would lie above it
  function automatic logic [KW-1:0] sub_of(input logic [BAW-1:0] b);
    logic [PCB-1:0] pc;
    pc = PCB'(b >> G) ^ (PC_XOR ? PCB'(b >> (G + PCB)) : PCB'(0));
    return (NS == 1) ? '0 : KW'(pc >> (PCB - KW));
  endfunction

  // ================================================================ native ports: input stage F, sub-port selection
  // (250 MHz uclk, FK35: c_cmd_* -> the sub-port's write FIFO / CAM / CQ were 4-11 levels from
  // oc_reg of otpu_mem_ch, 2.1-2.5 ns of route into 132 loads, and c_cmd_ready was a chain of 8-10
  // levels from h_we / ws_ok / urst into a 672-pin write enable: oc_reg -> a_rdep -0.25 ns,
  // h_we / ws_ok / calib_u2 -> u_wq -0.19 ns.) Now every input of c_* goes into a register F, free-running
  // (the command, the write data), with no enable; the only logic on the way is c_cmd_ready, a LUT of
  // registers (rrd / rwr: credits, below) and c_cmd_we / c_wdata_valid, and f_v / f_wp, a LUT of that
  // and the command. F lives for one cycle: its command goes into the queue CQ of its sub-port, its
  // write data into WQ, its read's sub-port number into the FIFO tq -- nothing there can refuse it,
  // because the credits kept room. A command accepted in cycle t is in F in t + 1 and pushed at the
  // end of it. rrd / rwr (registers, for cycle t + 1) are set in cycle t only if in every sub-port of
  // the native port the occupancy counter (cq_n / wq_n: pushes counted when they happen, pops one
  // cycle late, so never below the queue's fill) plus the push from F in t plus 2 still fits: 2 for
  // the command accepted in t (whichever sub-port it goes to: not known without the long route from
  // otpu_mem_ch) and the one that may be accepted in t + 1.
  logic [1:0][KW-1:0] sel;                  // sub-port of the command on c_cmd
  logic [NA-1:0]      cq_ok, wq_ok;         // room in the sub-port's CQ / WQ for 2 more commands and the one in F
  logic [1:0]         rrd, rwr;             // c_cmd_ready of a read / a write (registers: credits)
  logic [1:0]         f_v, f_we;            // F: a command was accepted in the previous cycle, its we
  logic [1:0][BAW-1:0] f_addr;              // F: its beat
  logic [1:0][KW-1:0] f_sel;                // F: its sub-port
  logic [1:0][DW-1:0] f_wd;                 // F: the write data (every cycle, valid with f_v && f_we)
  logic [1:0][NS-1:0][NG-1:0] f_wp;         // F: a write for the sub-port: its WQ's write enable (NG copies)
  logic [1:0]         tq_rv;                // FIFO of read sub-port numbers: not empty
  logic [1:0][KW-1:0] tq_rd;                // sub-port of the next read beat of the native port
  logic [NA-1:0]      hr;                   // the head of the sub-port's read ring is ready (both words)
  (* max_fanout = 32 *) logic [NA-1:0] pk;  // its beat goes to otpu_mem_ch in this cycle
  logic [NA-1:0][511:0] q;                  // the beat read from the sub-port's buffer at issue
  logic [1:0]         hn, pkn;              // native port: head ready / issues a beat
  logic [NA-1:0]      e_rr, e_br, e_prot;   // errors of this cycle: RRESP, BRESP, protocol
  logic [1:0]         ck_wnd;               // (simulation) a write command without data

  for (genvar p = 0; p < 2; p++) begin : g_n
    logic        l_fv, l_fwe, l_rrd, l_rwr, rd_ok, wr_ok, acc;
    logic [BAW-1:0] l_fa;
    logic [KW-1:0] l_fs;
    logic [DW-1:0] l_fd;
    (* keep = "true" *) logic [NS-1:0][NG-1:0] l_fwp;   // keep: the copies are not merged back into one
    assign sel[p] = sub_of(c_cmd_addr[p]);
    // a write is taken only with its data (and the room for both), a read with room for the command;
    // the credits do not depend on the command's sub-port: the command waits if either is short
    assign c_cmd_ready[p]   = c_cmd_we[p] ? (rwr[p] && c_wdata_valid[p]) : rrd[p];
    assign c_wdata_ready[p] = c_cmd_valid[p] && c_cmd_ready[p] && c_cmd_we[p];
    assign acc = c_cmd_valid[p] && c_cmd_ready[p];
    assign ck_wnd[p] = c_cmd_valid[p] && c_cmd_we[p] && !c_wdata_valid[p];
    always_comb begin
      rd_ok = 1'b1;
      wr_ok = 1'b1;
      for (int k = 0; k < NS; k++) begin
        rd_ok = rd_ok && cq_ok[p + 2 * k];
        wr_ok = wr_ok && wq_ok[p + 2 * k];
      end
    end
    // urst: a command accepted in the first cycle of a reset is dropped here (the queues' own reset
    // acts at the end of that cycle, before F would push it); rrd / rwr are 0 from the second cycle
    // of the reset on
    always_ff @(posedge uclk) begin
      l_fv  <= !urst && acc;
      l_fwe <= c_cmd_we[p];
      l_fa  <= c_cmd_addr[p];
      l_fs  <= sel[p];
      l_fd  <= {c_wdata_we[p], c_wdata_data[p]};
      for (int k = 0; k < NS; k++)
        for (int g = 0; g < NG; g++)
          l_fwp[k][g] <= !urst && acc && c_cmd_we[p] && sel[p] == KW'(k);
      l_rrd <= !urst && rd_ok;
      l_rwr <= !urst && rd_ok && wr_ok;
    end
    assign f_v[p]    = l_fv;
    assign f_we[p]   = l_fwe;
    assign f_addr[p] = l_fa;
    assign f_sel[p]  = l_fs;
    assign f_wd[p]   = l_fd;
    assign f_wp[p]   = l_fwp;
    assign rrd[p]    = l_rrd;
    assign rwr[p]    = l_rwr;
    // the sub-port number of every read, in the order of the native port's commands: pushed from F
    // (the credits keep room: at most 1 + NS * (CQ + 2 + RD) reads of a native port are accepted and
    // not yet delivered -- F, then per sub-port CQ, A, H and the ring -- and TQ is 4 * NS * RD >= that
    // for RD >= 4), popped when its beat is issued to otpu_mem_ch
    if (NS > 1) begin : g_tq
      otpu_hbm_fifo #(.W(KW), .DEPTH(TQ), .CRED(1'b1)) u_tq (.clk(uclk), .rst(urst),
        .wvalid(f_v[p] && !f_we[p]), .wready(),
        .wdata(f_sel[p]), .rvalid(tq_rv[p]), .rready(pkn[p]), .rdata(tq_rd[p]));
    end else begin : g_ntq
      assign tq_rv[p] = 1'b1;
      assign tq_rd[p] = '0;
    end
    always_comb begin
      hn[p] = 1'b0;
      for (int k = 0; k < NS; k++) if (tq_rd[p] == KW'(k)) hn[p] = tq_rv[p] && hr[p + 2 * k];
    end
  end

  // ================================================================ AXI ports (sub-ports)
  for (genvar a = 0; a < NA; a++) begin : g_p
    localparam int p = a % 2;               // native port
    localparam int k = a / 2;               // sub-port
    // ---------------------------------------------------------------- operation tables
    logic [RD-1:0]        r_pend;           // read in the HBM (for the CAM)
    logic [RD-1:0]        r_done;           // both words are in the buffer, waiting to be issued
    logic [RD-1:0]        r_half;           // the first word has been received
    logic [RD-1:0][HW-1:0] r_beat;
    logic [RSW-1:0]       r_wp, r_rp, r_rp1;   // ring of read slots: AR issue, data issue, data issue + 1
    (* keep = "true" *) logic [NG-1:0][RSW-1:0] r_rpg;   // copies of r_rp: the read address of slice g of the data RAMs
    logic                 hr0;              // r_done[r_rp]: the head of the ring is ready (a register, see hr0 below)
    logic [RSW:0]         r_cnt;            // occupied slots
    logic [RD-1:0]        r_fr;             // the read completed in the previous cycle (onehot)
    logic [WD-1:0]        w_v;              // write in the HBM (until B)
    logic [WD-1:0][HW-1:0] w_beat;
    logic [WSW-1:0]       w_wp, w_wp1;      // ring of write slots: AW issue, AW issue + 1
    logic [WD-1:0]        w_fr;             // B arrived in the previous cycle (onehot)
    logic                 rs_ok, ws_ok;     // the next read / write slot is free

    // ---------------------------------------------------------------- CQ, A and H
    // CQ: the sub-port's commands, {we, beat}, pushed from F (credits: it never overflows), popped
    // into A. The head has copies (the CAM's comparators, A): one kept register set per copy
    logic                 cq_wv, cq_rv, cq_we;
    logic [2*NCP:0][BAW:0] cq_h;            // head copies: [0, NCP) read table, [NCP, 2 NCP) write table, [2 NCP] A
    logic [BAW-1:0]       cq_beat;
    logic                 a_v, a_we, h_v, h_we, h_ok;
    logic [BAW-1:0]       a_beat, h_beat;   // (the CAM compares their low HW bits; amap gets all BAW)
    // copies of h_beat for the entries of each table (250 MHz uclk, FK35: h_beat -> w_beat / r_beat,
    // 0 levels, 98% route, -0.43 ns: one register drove 64 table registers over 4.1 ns of route)
    (* keep = "true" *) logic [NHB-1:0][HW-1:0] h_beat_r, h_beat_w;
    logic [RD-1:0]        a_rdep, h_rdep, c_rm;
    logic [WD-1:0]        a_wdep, h_wdep, c_wm;
    (* max_fanout = 32 *) logic a_ld;       // the command at the head of CQ moves into A
    logic                 h_go, a_mv, al_r, al_w, eq_c, eq_a;
    logic [RD-1:0]        al_rv, a_rdep_n;
    logic [WD-1:0]        al_wv, a_wdep_n;
    logic                 ar_wr, aw_wr;     // space in the AR, AW FIFOs (registers)
    logic [32:0]          h_addr;

    assign h_go  = h_v && h_ok && (h_we ? (aw_wr && ws_ok) : (ar_wr && rs_ok));
    assign a_mv  = a_v && (!h_v || h_go);
    assign a_ld  = cq_rv && (!a_v || a_mv);
    assign al_r  = h_go && !h_we;
    assign al_w  = h_go && h_we;
    assign al_rv = RD'(al_r) << r_wp;
    assign al_wv = WD'(al_w) << w_wp;
    assign cq_we   = cq_h[2*NCP][BAW];
    assign cq_beat = cq_h[2*NCP][BAW-1:0];
    assign eq_c  = h_beat[HW-1:0] == cq_beat[HW-1:0];
    assign eq_a  = h_beat[HW-1:0] == a_beat[HW-1:0];

    // CAM: outstanding operations of the sub-port with the beat of the command moving into A (from
    // the CQ head's copy of its table's slice: oc_reg -> a_rdep was 4 levels, 2.1 ns of route, -0.25 ns)
    always_comb begin
      for (int e = 0; e < RD; e++) c_rm[e] = r_pend[e] && r_beat[e] == cq_h[e * NCP / RD][HW-1:0];
      for (int e = 0; e < WD; e++) c_wm[e] = w_v[e] && w_beat[e] == cq_h[NCP + e * NCP / WD][HW-1:0];
    end
    // dependency vector of A in the next cycle (if A does not accept a new command): completed ones
    // clear, the operation of the same beat just issued by the head is added
    assign a_rdep_n = (a_rdep & ~r_fr) | ((eq_a && a_we) ? al_rv : '0);
    assign a_wdep_n = (a_wdep & ~w_fr) | (eq_a ? al_wv : '0);

    always_ff @(posedge uclk) begin
      if (urst) a_v <= 1'b0;
      else if (a_ld) a_v <= 1'b1;
      else if (a_mv) a_v <= 1'b0;
      if (a_ld) begin
        a_we   <= cq_we;
        a_beat <= cq_beat;
        a_rdep <= cq_we ? ((c_rm & ~r_fr) | (eq_c ? al_rv : '0)) : '0;
        a_wdep <= (c_wm & ~w_fr) | (eq_c ? al_wv : '0);
      end else begin
        a_rdep <= a_rdep_n;
        a_wdep <= a_wdep_n;
      end

      if (urst) h_v <= 1'b0;
      else if (a_mv) h_v <= 1'b1;
      else if (h_go) h_v <= 1'b0;
      if (a_mv) begin
        h_we   <= a_we;
        h_beat <= a_beat;
        for (int j = 0; j < NHB; j++) begin
          h_beat_r[j] <= a_beat[HW-1:0];
          h_beat_w[j] <= a_beat[HW-1:0];
        end
        h_rdep <= a_rdep_n;
        h_wdep <= a_wdep_n;
        h_ok   <= !(|(a_rdep & ~r_fr)) && !(|(a_wdep & ~w_fr)) &&
                  !(eq_a && (al_w || (al_r && a_we)));
      end else begin
        h_rdep <= h_rdep & ~r_fr;
        h_wdep <= h_wdep & ~w_fr;
        h_ok   <= !(|(h_rdep & ~r_fr)) && !(|(h_wdep & ~w_fr));
      end
    end

    otpu_hbm_amap #(.CH_BASE(CH_BASE), .PCB(PCB), .BAW(BAW), .PC_GRAN(PC_GRAN), .PC_XOR(PC_XOR)) u_amap (
      .beat(h_beat), .addr(h_addr));

    // ---------------------------------------------------------------- CQ and the credits
    // F -> CQ in the cycle after the accept; the head of CQ is A's input (copies of the head, see above)
    assign cq_wv = f_v[p] && f_sel[p] == KW'(k);
    otpu_hbm_fifo #(.W(BAW + 1), .DEPTH(CQ), .CRED(1'b1), .HC(2 * NCP + 1)) u_cq (.clk(uclk), .rst(urst),
      .wvalid(cq_wv), .wready(), .wdata({f_we[p], f_addr[p]}), .rvalid(cq_rv), .rready(a_ld),
      .rdata(cq_h));
    // Occupancy counters of CQ and WQ for the credits of F (above): the pushes from F, the pops one
    // cycle late (a_ld / the WQ pop registered), so a counter is never below the queue's fill. A
    // command accepted in cycle t is in F in t + 1 and pushed at its end, so room for it in cycle t + 1
    // means: count + the command in F now + the one accepted now (any sub-port, not known without a
    // long route from otpu_mem_ch) + the one that may be accepted in t + 1 <= the depth
    logic [$clog2(CQ):0]  cq_n;
    logic [$clog2(WQ):0]  wq_n;
    logic                 cq_pop_r, wq_pop_r, wq_pop;
    assign cq_ok[a] = int'(cq_n) + int'(cq_wv) + 2 <= CQ;
    assign wq_ok[a] = int'(wq_n) + int'(f_wp[p][k][0]) + 2 <= WQ;
    always_ff @(posedge uclk) begin
      if (urst) begin
        cq_n <= '0; wq_n <= '0; cq_pop_r <= 1'b0; wq_pop_r <= 1'b0;
      end else begin
        cq_n     <= cq_n + ($clog2(CQ) + 1)'(cq_wv) - ($clog2(CQ) + 1)'(cq_pop_r);
        wq_n     <= wq_n + ($clog2(WQ) + 1)'(f_wp[p][k][0]) - ($clog2(WQ) + 1)'(wq_pop_r);
        cq_pop_r <= a_ld;
        wq_pop_r <= wq_pop;
      end
    end

    // ---------------------------------------------------------------- AR, AW
    logic [QW-1:0] ar_rd, aw_rd;
    otpu_hbm_fifo #(.W(QW), .DEPTH(AQ)) u_arq (.clk(uclk), .rst(urst), .wvalid(al_r),
      .wready(ar_wr), .wdata({h_addr, 6'(r_wp)}), .rvalid(m_arvalid[a]), .rready(m_arready[a]),
      .rdata(ar_rd));
    otpu_hbm_fifo #(.W(QW), .DEPTH(AQ)) u_awq (.clk(uclk), .rst(urst), .wvalid(al_w),
      .wready(aw_wr), .wdata({h_addr, 6'(w_wp)}), .rvalid(m_awvalid[a]), .rready(m_awready[a]),
      .rdata(aw_rd));
    assign m_araddr[a]  = ar_rd[QW-1 -: 33];
    assign m_arid[a]    = ar_rd[5:0];
    assign m_arlen[a]   = 4'd1;
    assign m_arsize[a]  = 3'd5;
    assign m_arburst[a] = 2'b01;
    assign m_awaddr[a]  = aw_rd[QW-1 -: 33];
    assign m_awid[a]    = aw_rd[5:0];
    assign m_awlen[a]   = 4'd1;
    assign m_awsize[a]  = 3'd5;
    assign m_awburst[a] = 2'b01;

    // ---------------------------------------------------------------- W
    // the write data in the order of the sub-port's write commands = in the order of AW; a packet starts only
    // when its AW has already been accepted by the HBM (w_cred: AWs accepted minus started W packets)
    logic          wq_rv, wq_rr, wo_v, wo_h, w_start, aw_hs;
    logic [DW-1:0] wq_rd;
    logic [$clog2(WQ)+1:0] w_cred;
    logic [WSW-1:0] w_idc;
    logic [255:0]  wo_data;
    logic [31:0]   wo_strb, wo_par;
    logic          wo_last;
    logic [5:0]    wo_id;
    // pushed from F with the command (registered write enables, NG copies for the NG slices of the
    // RAM; the credits keep room). 250 MHz uclk, FK35: h_we / ws_ok / calib_u2 -> the accept -> the
    // 672 write enables (8-10 levels, 336-496 endpoints, -0.16 to -0.23 ns)
    otpu_hbm_fifo #(.W(DW), .DEPTH(WQ), .CRED(1'b1), .NG(NG)) u_wq (.clk(uclk), .rst(urst),
      .wvalid(f_wp[p][k]), .wready(), .wdata(f_wd[p]), .rvalid(wq_rv), .rready(wq_rr),
      .rdata(wq_rd));
    assign wq_pop  = wq_rv && wq_rr;
    assign aw_hs   = m_awvalid[a] && m_awready[a];
    assign w_start = (!wo_v || m_wready[a]) && !wo_h && wq_rv && w_cred != 0;
    assign wq_rr   = (!wo_v || m_wready[a]) && wo_h;
    always_ff @(posedge uclk) begin
      if (urst) begin
        wo_v <= 1'b0; wo_h <= 1'b0; w_cred <= '0; w_idc <= '0;
      end else begin
        w_cred <= w_cred + ($clog2(WQ) + 2)'(aw_hs) - ($clog2(WQ) + 2)'(w_start);
        if (!wo_v || m_wready[a]) begin
          wo_v <= w_start || wo_h;
          wo_h <= w_start;
          if (wo_h) w_idc <= w_idc + 1'b1;
        end
      end
      if (!wo_v || m_wready[a]) begin
        wo_data <= wo_h ? wq_rd[511:256] : wq_rd[255:0];
        wo_strb <= wo_h ? wq_rd[575:544] : wq_rd[543:512];
        wo_last <= wo_h;
        wo_id   <= 6'(w_idc);
        for (int i = 0; i < 32; i++)
          wo_par[i] <= ^(wo_h ? wq_rd[256 + 8 * i +: 8] : wq_rd[8 * i +: 8]);
      end
    end
    assign m_wvalid[a]       = wo_v;
    assign m_wdata[a]        = wo_data;
    assign m_wstrb[a]        = wo_strb;
    assign m_wlast[a]        = wo_last;
    assign m_wid[a]          = wo_id;
    assign m_wdata_parity[a] = wo_par;

    // ---------------------------------------------------------------- B
    logic          bb_v;
    logic [5:0]    bb_id;
    logic [1:0]    bb_resp;
    assign m_bready[a] = 1'b1;
    always_ff @(posedge uclk) begin
      bb_v    <= !urst && m_bvalid[a];
      bb_id   <= m_bid[a];
      bb_resp <= m_bresp[a];
    end

    // ---------------------------------------------------------------- R
    // The word's registers (rr_*) take the HBM IP's outputs as before. The read buffer's RAMs are
    // written from copies of the address and the enables, one set per slice (rr_sg, rr_vg, rr_lg: kept
    // registers, NG slices of 64 bits of each word), and the word's half comes from RLAST (the second
    // word is the last: its half is r_hi), not from the slot's half flag r_half behind a 32:1 mux
    // (250 MHz uclk, FK35: rr_id -> r_lo / r_hi write address, 0 levels, 98% route, 613 loads,
    // -0.23 ns; rr_id -> the write enables, 4 levels, -0.08 ns; rr_id -> r_fr / r_done, 4-5 levels,
    // -0.06 ns). r_half is only the protocol check now: the word of an ID is the first or the second
    // one in turn, and RLAST must agree with it (e_prot).
    logic          rr_v, rr_last;
    logic [5:0]    rr_id;
    logic [1:0]    rr_resp;
    logic [255:0]  rr_data;
    logic [RSW-1:0] rr_s;
    (* keep = "true" *) logic [NG-1:0]          rr_vg, rr_lg;
    (* keep = "true" *) logic [NG-1:0][RSW-1:0] rr_sg;
    logic [511:0]  qd;
    assign m_rready[a] = 1'b1;
    always_ff @(posedge uclk) begin
      rr_v    <= !urst && m_rvalid[a];
      rr_id   <= m_rid[a];
      rr_resp <= m_rresp[a];
      rr_last <= m_rlast[a];
      rr_data <= m_rdata[a];
      for (int g = 0; g < NG; g++) begin
        rr_vg[g] <= m_rvalid[a];            // (a write of garbage during a reset is harmless: read only after r_done)
        rr_lg[g] <= m_rlast[a];
        rr_sg[g] <= m_rid[a][RSW-1:0];
      end
    end
    assign rr_s = rr_id[RSW-1:0];
    for (genvar g = 0; g < NG; g++) begin : g_rm
      localparam int SWR = 256 / NG;
      (* ram_style = "distributed" *) logic [SWR-1:0] r_lo [RD];
      (* ram_style = "distributed" *) logic [SWR-1:0] r_hi [RD];
      logic [SWR-1:0] lo_q, hi_q;
      always_ff @(posedge uclk) begin
        if (rr_vg[g] && !rr_lg[g]) r_lo[rr_sg[g]] <= rr_data[g * SWR +: SWR];
        if (rr_vg[g] &&  rr_lg[g]) r_hi[rr_sg[g]] <= rr_data[g * SWR +: SWR];
        // the head of the ring is read every cycle (no pk fanout to 512 bits): in the cycle
        // after the selection qd is the selected beat, r_rp advances on the same edge
        lo_q <= r_lo[r_rpg[g]];
        hi_q <= r_hi[r_rpg[g]];
      end
      assign qd[g * SWR +: SWR]       = lo_q;
      assign qd[256 + g * SWR +: SWR] = hi_q;
    end

    // ---------------------------------------------------------------- table state
    logic [WD-1:0] w_v_n;
    logic [6:0]   r_inf, w_inf;
    logic         ev, ev0, ev1;
    assign ev      = rr_v && rr_last;       // the last word of a read has arrived
    assign ev0     = ev && rr_s == r_rp;    // ... for the head of the ring
    assign ev1     = ev && rr_s == r_rp1;   // ... for the slot after it
    assign hr[a]   = hr0;
    assign pk[a]   = pkn[p] && tq_rd[p] == KW'(k);
    assign q[a]    = qd;
    assign w_v_n   = (w_v & ~w_fr) | al_wv;
    // hr0 == r_done[r_rp], kept as a register: the head's readiness was a 32:1 mux of r_done by r_rp
    // in front of hn -> pkn -> pk -> the tq FIFO's pointers (250 MHz uclk, FK35: r_rp -> u_tq rv / wr,
    // 11 levels, -0.35 ns). After a pop the next head is the slot r_rp1 (r_done there, or the word
    // arriving now); without a pop the head's flag only gains the arrival. rs_ok / ws_ok likewise:
    // the late signals (al_r, pk, al_w) only select between values computed from registers. A slot
    // is full (cnt == RD) until a pop; it becomes full with an issue and no pop at cnt == RD - 1
    always_ff @(posedge uclk) begin
      if (urst) begin
        r_pend <= '0; r_done <= '0; r_half <= '0; r_wp <= '0; r_rp <= '0; r_cnt <= '0;
        r_rp1 <= RSW'(1); r_rpg <= '0; hr0 <= 1'b0;
        r_fr <= '0; rs_ok <= 1'b1;
        w_v <= '0; w_wp <= '0; w_wp1 <= WSW'(1); w_fr <= '0; ws_ok <= 1'b1;
        r_inf <= '0; w_inf <= '0;
      end else begin
        r_pend <= (r_pend & ~r_fr) | al_rv;
        r_fr   <= ev ? RD'(1) << rr_s : '0;
        r_half <= r_half ^ (rr_v ? RD'(1) << rr_s : '0);
        r_done <= (r_done | (ev ? RD'(1) << rr_s : '0)) &
                  ~(pk[a] ? RD'(1) << r_rp : '0);
        hr0    <= pk[a] ? (r_done[r_rp1] || ev1) : (hr0 || ev0);
        if (al_r)  r_wp <= r_wp + 1'b1;
        if (pk[a]) begin
          r_rp  <= r_rp1;
          r_rp1 <= r_rp1 + 1'b1;
          for (int g = 0; g < NG; g++) r_rpg[g] <= r_rpg[g] + 1'b1;
        end
        r_cnt <= (al_r && !pk[a]) ? r_cnt + 1'b1 : (pk[a] && !al_r) ? r_cnt - 1'b1 : r_cnt;
        rs_ok <= pk[a] || (rs_ok && !(al_r && r_cnt == (RSW + 1)'(RD - 1)));
        w_v   <= w_v_n;
        w_fr  <= bb_v ? WD'(1) << bb_id[WSW-1:0] : '0;
        if (al_w) begin
          w_wp  <= w_wp1;
          w_wp1 <= w_wp1 + 1'b1;
        end
        ws_ok <= al_w ? !(w_v[w_wp1] && !w_fr[w_wp1]) : !(w_v[w_wp] && !w_fr[w_wp]);
        r_inf <= r_inf + 7'(al_r) - 7'(ev);
        w_inf <= w_inf + 7'(al_w) - 7'(bb_v);
      end
      // the table entries take h_beat from their slice's copy (h_beat_r / h_beat_w)
      for (int e = 0; e < RD; e++) if (al_r && r_wp == RSW'(e)) r_beat[e] <= h_beat_r[e * NHB / RD];
      for (int e = 0; e < WD; e++) if (al_w && w_wp == WSW'(e)) w_beat[e] <= h_beat_w[e * NHB / WD];
    end

    // errors of this cycle: a response != OKAY; R/B with an ID without an operation, RLAST not on the second word
    // (the protocol check one cycle later, from registered lookups: it only feeds st_perr)
    logic k_rv, k_ridbad, k_pend, k_done, k_half, k_last, k_bv, k_bidbad, k_bwv;
    always_ff @(posedge uclk) begin
      k_rv     <= !urst && rr_v;
      k_ridbad <= {1'b0, rr_id} >= 7'(RD);
      k_pend   <= r_pend[rr_s];
      k_done   <= r_done[rr_s];
      k_half   <= r_half[rr_s];
      k_last   <= rr_last;
      k_bv     <= !urst && bb_v;
      k_bidbad <= {1'b0, bb_id} >= 7'(WD);
      k_bwv    <= w_v[bb_id[WSW-1:0]];
    end
    assign e_rr[a]   = rr_v && rr_resp != 2'b00;
    assign e_br[a]   = bb_v && bb_resp != 2'b00;
    assign e_prot[a] = (k_rv && (k_ridbad || !k_pend || k_done || k_last != k_half)) ||
                       (k_bv && (k_bidbad || !k_bwv));

    // ---------------------------------------------------------------- counters
    logic [31:0] s_rcmd, s_wcmd, s_dep;
    logic        s_busy;
    always_ff @(posedge uclk) begin
      if (urst) begin
        s_rcmd <= '0; s_wcmd <= '0; s_dep <= '0; s_busy <= 1'b0;
      end else begin
        if (cq_wv && !f_we[p]) s_rcmd <= s_rcmd + 1'b1;
        if (cq_wv &&  f_we[p]) s_wcmd <= s_wcmd + 1'b1;
        if (h_v && !h_ok) s_dep <= s_dep + 1'b1;
        s_busy <= cq_wv || cq_rv || a_v || h_v || r_cnt != 0 || w_inf != 0 || wq_rv || wo_v ||
                  m_arvalid[a] || m_awvalid[a];
      end
    end
    assign st_rcmd[a] = s_rcmd;
    assign st_wcmd[a] = s_wcmd;
    assign st_dep[a]  = s_dep;
    assign st_busy[a] = s_busy;
    assign st_rout[a] = r_inf;
    assign st_wout[a] = w_inf;
  end

  // ================================================================ read data to otpu_mem_ch
  // round robin on a conflict: a beat leaves 3 cycles after the selection, valid of exactly one port.
  // The mux of the sub-ports' beats (o_a) is registered into rd_d, and c_rdata_* are a second register
  // stage after it, pure copies: placed between the two modules, they split the way into otpu_mem_ch's
  // read-data RAMs (250 MHz uclk, FK35: c_rdata_data_reg -> u_ar / u_xr RAM, 0 levels, 98% route,
  // 3.7 ns, -0.31 ns). The beat's slot was freed at the selection, so the slot's round trip is unchanged
  logic       last1;
  logic [1:0] o_v, o_v2;
  localparam int OAW = $clog2(NA);
  logic [OAW-1:0] o_a;                      // AXI port of the selected beat
  logic [511:0]   rd_d;
  assign pkn[0] = hn[0] && (!hn[1] || last1);
  assign pkn[1] = hn[1] && (!hn[0] || !last1);
  always_ff @(posedge uclk) begin
    if (urst) begin
      last1 <= 1'b0; o_v <= '0; o_v2 <= '0; c_rdata_valid <= '0;
    end else begin
      if (pkn[1]) last1 <= 1'b1;
      else if (pkn[0]) last1 <= 1'b0;
      o_v <= pkn;
      o_v2 <= o_v;
      c_rdata_valid <= o_v2;
    end
    o_a <= pkn[1] ? OAW'(1 + 2 * tq_rd[1]) : OAW'(2 * tq_rd[0]);
    rd_d <= q[o_a];
    c_rdata_data[0] <= rd_d;
    c_rdata_data[1] <= rd_d;
  end

  // ================================================================ errors
  logic [15:0] e_n;
  always_comb begin
    e_n = '0;
    for (int a = 0; a < NA; a++) e_n = e_n + 16'(e_rr[a]) + 16'(e_br[a]);
  end
  always_ff @(posedge uclk) begin
    if (urst) begin
      st_err_cnt <= '0; st_err <= 1'b0; st_perr <= 1'b0;
    end else begin
      if (e_n != 0) st_err_cnt <= (st_err_cnt > 16'hFFFF - e_n) ? 16'hFFFF : st_err_cnt + e_n;
      if (e_n != 0) st_err <= 1'b1;
      if (|e_prot) st_perr <= 1'b1;
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge uclk) if (!urst) begin
    for (int p = 0; p < 2; p++) begin
      if (c_cmd_valid[p] && c_cmd_addr[p][7] != 1'(p))
        $error("otpu_hbm_port: beat %0d on port %0d (bit 7)", c_cmd_addr[p], p);
      if (ck_wnd[p])
        $warning("otpu_hbm_port: port %0d: a write command without data on c_wdata", p);
      if (pkn[p] && !tq_rv[p])
        $error("otpu_hbm_port: port %0d: read data without a sub-port number", p);
    end
    if (&c_rdata_valid) $error("otpu_hbm_port: c_rdata_valid of both ports in the same cycle");
    if (|e_prot) $error("otpu_hbm_port: the HBM violated the protocol (R/B: ID, RLAST)");
  end
`endif
endmodule

// Mapping of the channel's beat number to the global HBM byte address (one-to-one).
// G = PC_GRAN, P = PCB (2^P pseudo-channels of 256 MiB in the channel): a block of 2^G adjacent beats lies
// in one pseudo-channel; the pseudo-channel number in the channel (address[27+P:28]) -- the bits beat[G+P-1:G], with
// PC_XOR also XORed with beat[G+2P-1:G+P]; the remaining beat bits in order -- the address inside the pseudo-channel
// (address[27:6]). A channel is 2^(22+P) beats; the beat bits above that are not used (BAW >= 22 + P: 25 for
// P = 2, 3 -- the default --, 26 for P = 4, where pc is address[31:28] and CH_BASE[32] is the HBM stack).
// Rationale: README.md.
module otpu_hbm_amap #(
  parameter logic [32:0] CH_BASE = 33'h0,
  parameter int  PCB     = 3,
  parameter int  BAW     = 25,
  parameter int  PC_GRAN = 0,
  parameter bit  PC_XOR  = 1'b1
) (
  input  logic [BAW-1:0] beat,
  output logic [32:0]    addr
);
  localparam int G = PC_GRAN;
  logic [PCB-1:0] pc;
  logic [21:0]    off;
  always_comb begin
    pc = PCB'(beat >> G);
    if (PC_XOR) pc = pc ^ PCB'(beat >> (G + PCB));
    off = 22'((beat >> (G + PCB)) << G) | 22'(beat & ((BAW'(1) << G) - BAW'(1)));
    addr = CH_BASE | (33'(pc) << 28) | (33'(off) << 6);
  end
endmodule

// Synchronous FIFO, first-word fall-through, distributed RAM; rvalid, wready and the head come from
// registers (like otpu_sfifo RO in otpu_mem_ch.sv), so that the AXI outputs and c_cmd_ready start at
// flip-flops. The late inputs (a push from the head's issue logic, a pop from the HBM's ready) only
// select between values computed from registers: the pointers, rv and wr are one LUT after them
// (250 MHz uclk, FK35: pkn -> u_tq pop -> rp_n -> rv / wr, 5 of 11 levels, -0.35 ns, had adders
// and compares of (wp_n - rp_n) behind them).
//   CRED: the writer has credits and never pushes into a full FIFO (no wr gating of the push; wready
//         is not used); a push into a full FIFO is an error in the simulation.
//   NG:   the RAM in NG slices of W / NG bits, each with its own copy of the write address and
//         enable (wvalid is NG equal copies, kept registers of the writer): one address register and one
//         enable drove all 576 bits of a write FIFO's RAM.
//   HC:   copies of the head register (rdata = HC copies of W bits, kept), for a head with a large fanout.
module otpu_hbm_fifo #(
  parameter int W = 8,
  parameter int DEPTH = 4,
  parameter bit CRED = 1'b0,
  parameter int NG = 1,
  parameter int HC = 1
) (
  input  logic            clk,
  input  logic            rst,
  input  logic [NG-1:0]   wvalid,
  output logic            wready,
  input  logic [W-1:0]    wdata,
  output logic            rvalid,
  input  logic            rready,
  output logic [HC*W-1:0] rdata
);
  localparam int AW = $clog2(DEPTH);
  localparam int SW = W / NG;
  initial begin
    if (DEPTH != (1 << AW) || DEPTH < 2) $fatal(1, "otpu_hbm_fifo: DEPTH = 2^k >= 2");
    if (W % NG != 0) $fatal(1, "otpu_hbm_fifo: W is not a multiple of NG");
  end
  logic [AW:0]   wp, rp, rp1;               // rp1 == rp + 1 (the head after a pop is read at it)
  logic          one, af, rv, wr, push, pop;
  logic [W-1:0]  mem_rd, nxt;
  assign push = CRED ? wvalid[0] : (wvalid[0] && wr);
  assign pop  = rv && rready;
  assign one  = wp == rp1;                                    // exactly one entry
  assign af   = (wp + 1'b1) == {~rp[AW], rp[AW-1:0]};         // DEPTH - 1 entries
  assign nxt  = one ? wdata : mem_rd;
  always_ff @(posedge clk) begin
    if (rst) begin
      wp <= '0; rp <= '0; rp1 <= (AW + 1)'(1); rv <= 1'b0; wr <= 1'b1;
    end else begin
      if (push) wp <= wp + 1'b1;
      if (pop) begin
        rp  <= rp1;
        rp1 <= rp1 + 1'b1;
      end
      rv <= push || (rv && !(pop && one));
      wr <= pop || (wr && !(push && af));
    end
  end
  for (genvar g = 0; g < NG; g++) begin : g_m
    (* ram_style = "distributed" *) logic [SW-1:0] mem [DEPTH];
    logic [AW-1:0] wa;
    if (NG > 1) begin : g_c
      (* keep = "true" *) logic [AW-1:0] wpc;                 // this slice's copy of wp
      always_ff @(posedge clk)
        if (rst) wpc <= '0;
        else if (wvalid[g] && (CRED || wr)) wpc <= wpc + 1'b1;
      assign wa = wpc;
    end else begin : g_s
      assign wa = wp[AW-1:0];
    end
    always_ff @(posedge clk) if (wvalid[g] && (CRED || wr)) mem[wa] <= wdata[g * SW +: SW];
    assign mem_rd[g * SW +: SW] = mem[rp1[AW-1:0]];
  end
  if (HC > 1) begin : g_h
    (* keep = "true" *) logic [HC-1:0][W-1:0] head;
    always_ff @(posedge clk)
      for (int c = 0; c < HC; c++)
        if (pop) head[c] <= nxt;
        else if (!rv) head[c] <= wdata;
    assign rdata = head;
  end else begin : g_h
    logic [W-1:0] head;
    always_ff @(posedge clk)
      if (pop) head <= nxt;
      else if (!rv) head <= wdata;
    assign rdata = head;
  end
  assign rvalid = rv;
  assign wready = wr;
`ifndef SYNTHESIS
  always_ff @(posedge clk) if (!rst) begin
    if (CRED && wvalid[0] && wp == {~rp[AW], rp[AW-1:0]})
      $error("otpu_hbm_fifo: push into a full FIFO (W = %0d, DEPTH = %0d): the credits are wrong", W, DEPTH);
    if (NG > 1 && wvalid != '0 && wvalid != '1) $error("otpu_hbm_fifo: the copies of wvalid differ");
  end
`endif
endmodule
