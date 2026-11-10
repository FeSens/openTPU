// A behavioural model of one DDR4 DIMM's AXI4 port on the BCU1525 at WG 8 (simulation only:
// tb_bcuwg): what otpu_mig_pair's m_* sees through the block design's SmartConnect and the DIMM's
// MIG, 512-bit beats (AxSIZE 64 bytes, INCR), IDW-bit IDs, byte addresses of DAW bits (CL + log2 NC:
// 30 at 512 MiB a channel, 33 at 4 GiB). The memory is the DIMM's NC virtual channels (board
// channels CH0 .. CH0 + NC - 1, CH0 2 DIMM at two a DIMM: the DIMM's byte {o[CL-1:12], s, o[11:0]}
// is channel s's byte o, otpu_mig_pair's map; at two slices, four a DIMM, the slot's), each its
// first BEATS 32-byte beats in an array (the image) and every beat above them that is loaded or
// written in an associative one (the channel's 2^CL bytes in all).
// As otpu_hbm_model, the freedom AXI gives the slave between its read and write channels: a read
// takes its data at a random cycle between its AR and its first R beat, a write lands at a random
// cycle between its last W beat and its B, so a read and a write in flight together on one
// address may pass each other either way. Writes land in AW order and their B's come in that
// order; reads return in AR order per ID, and +dimm_ooo=P: when the head read's data is not due
// yet, a later one of another ID, due, goes first in P percent of the cycles (the MIG's ordering
// across IDs; no interleaving inside a burst). BID / RID echo AWID / ARID.
//
// Timing (clk cycles; aclk, 300 MHz): +dimm_lat=N from an AR to its first R beat (default 75:
// 250 ns), +dimm_wlat=N from a last W beat to its B (default 40), +dimm_jit=N up to N more on
// each; +dimm_bw=P: the DIMM's data bus, shared by R and W, carries a beat in P percent of the
// cycles at most (refresh, page misses, turnarounds; default 95); +dimm_st=P: ARREADY / AWREADY /
// WREADY low in P percent of the cycles. ARQ read and AWQ write bursts in flight at most.
//
// +dimm_mc=1: the timing is the MIG's instead (bd_wg.tcl's ddr4 IP: PG150's AXI front end, user
// interface and controller as its generated RTL schedules, docs/bcu1525_wg.md), on DDR4-2400
// UDIMM timings (tCK 833 ps, four a clk cycle). The front end splits a burst into 64-byte
// commands, one a cycle, reads and writes by RD_PRI_REG: a direction holds while its commands come
// and another's wait, and gives way at a burst's end after 254 commands (ddr4_v2_2_axi_cmd_arbiter);
// a write's commands go once its W is in. The user interface takes them in order, each into its
// bank group's transaction FIFO (two at most: TXN_FIFO_FULL_THRESHOLD) while fewer than 32 are
// between it and their CAS (DBAW 5). Each group (ddr4_v2_2_mc_group) takes its head in order: a
// page hit into its CAS FIFO (two at most); a miss once its CASes are out, a precharge, tRP, an
// activate, the CAS. One CAS a cycle (the direction of the last first), one ACT and one PRE, under
// tRCD, tRP, tRAS, tRTP, tWR, tRRD_S / L, tFAW, tCCD_L, tRTW and tWTR_S / L; a refresh every tREFI
// (the user interface held, the groups drained, every bank closed for tRP + tRFC). The address:
// ROW_COLUMN_BANK, the bank group's bits at byte bits +mc_bg=N and +mc_bg1=N (default 6 and 7)
// and the bank's at +mc_ba=N and +mc_ba1=N (8 and 9; each second one defaults to the first + 1),
// the column the next seven bits up and the row the rest, up to bit DAW - 1 (an address below
// 2^30 has the row it had at DAW 30): an address map in front of the MIG's. +mc_wbg=N, +mc_wbg1=N,
// +mc_wba=N and +mc_wba1=N (card-gap; default off): the bank-half window's DIMM bytes (below
// 2^(BHW + log2 NC)) take their own group and bank bits (the second ones default as above, +mc_wba
// to +mc_ba's), the rest +mc_bg / +mc_ba's: a split map (the row bits the same in both: a bijection).
// +mc_bgx=N (card-gap; default off): outside the window's map the group's high bit is +mc_bg1's
// XOR byte bit N's (N one of the map's other bits: still a bijection).
// A read takes its data at its CAS and gives it +mc_rl=N cycles later (default 32: CL, the PHY,
// the read buffer, SmartConnect); a burst reaches the front end +mc_al=N cycles after its
// handshake (default 14); a write lands at its CAS and answers +mc_bl=N cycles after its last
// command goes in (default 8). W beats are taken while fewer than 48 wait for their CAS. Writes
// still in the controller land at the dump.
// +mc_rw=N: the front end gives a read command only while fewer than N reads are between their
// command and their data's beat leaving the MIG's AXI port (ddr4_v2_2_axi_r_channel's r_data_rdy:
// its transaction FIFO of 30, the read data FIFO of 32; the user interface's read buffer of 32
// under ORDERING Normal), the beat leaving mc_rl - mc_rs cycles after its CAS, in AR order, one a
// cycle, and reaching R +mc_rs=N cycles later (SmartConnect); default 32 and 0, +mc_rw=0 no window.
// With the defaults and +mc_ba=15 the model is f150m15's within 1.5% (docs/bcu1525_wg.md: its
// pointer chase, its int8 and fp4 streams, a DeltaNet layer); under the built map its fp4 stream is
// 9% long (its scales' reads beside the stream: the bank conflicts too dear), the int8 2%.
// +mc_strict=1: the CASes in the order the user interface took their commands (ORDERING Strict,
// ddr4_v2_2_mc_arb_c's strict FIFO) instead of the groups' round robin.
// +mc_sc=1 (with +mc_rw): the SmartConnect's R path as its funcsim netlist runs it (f150r62's
// sc_mem0 between an ideal in-order memory and a master always ready, docs/bcu1525_wg.md): a
// burst of L beats takes L + 1 cycles, one idle R cycle after each RLAST whatever the IDs or the
// clocks, and the beats leave the MIG's port only while fewer than +mc_scq=N (default 32: its R
// node's FIFO) are between it and R.
// Calibration knobs against the card (tCK, default 0: the timings below): +mc_xrtw=N and +mc_xwtr=N
// N more on the read-to-write and write-to-read turnarounds; +mc_xbsw=N a CAS to another bank than
// its group's last CAS waits N after that CAS (a bank switch inside a group); +mc_xrcd=N N more on
// tRCD; +mc_xbq=1 a page hit to another bank than the group's CAS FIFO holds waits for the FIFO to
// empty, as a miss does (a group queue in one bank at a time); +mc_wnl=1 the front end's write
// direction holds while writes come, without RD_PRI_REG's 256 commands; +mc_wrate=N a write CAS
// at most every N clk cycles (1: every cycle); +mc_xccd=N N more on tCCD_L (a group's CAS after its
// last; +mc_xccdb=1 only after a CAS of the same burst, the front end's gap between bursts
// spacing the rest; +mc_xccdb=2 only after a CAS of the same channel); +mc_xgq=N a group moves a
// transaction into its CAS FIFO at most every N (its pipeline: a rate, the FIFO's two entries
// absorbing a close pair, where +mc_xccd spaces every pair).
// The card's calibration (K7, 11-02): +mc_xbsw=16,+mc_xrtw=8,+mc_xwtr=8,+mc_xgq=12, decode within
// 1.11% of the BCU1525's at one slice (pbhw16, b15) and two (s2r150): q36-4, q38-4 and goss-2. The
// split map s16 at one slice it puts +9.4% over b15, the card +12.0%.
// +mc_actlog=PREFIX (with +dimm_mc=1): each ACT a line in PREFIXd<DIMM>.txt (appended, a "#" line
// opening each simulation): the cycle, the group, the bank, the row it opens and the row the bank
// held, why the bank was closed (P: a precharge for this row, R: a refresh, C: never open), then
// the command it opens for and the bank's last CAS before it closed, each as write (1) or read,
// channel and channel byte (before the window's swap). The timing is the same with it
// (tools/actbin.py bins the ACTs by cause).
//
// Checks: a valid held until its ready with the same payload (AR, AW, W); addresses 64-byte
// aligned, a burst of 16 beats at most (the pairs': 8) inside 4 KiB; WLAST on the burst's last
// beat; W only after its AW (AXI4: W in AW order).
//
// Images (PHYS, as otpu_hbm_model's), for c = CH0 .. CH0 + NC - 1, +dir=<d>: the first BEATS beats
// <d>/ch<c>.bin (32-bit big-endian words, as $fread reads them; shorter or absent: zeros) in at
// time 0 and <d>/ch<c>_out.bin (BEATS x 32 bytes, little-endian words) out at the rising edge that
// sees dump; the beats above them <d>/ch<c>_hi.txt in (absent: none; a beat not in it reads 0) and
// <d>/ch<c>_hi_out.txt out, every beat the model holds there (loaded or written). Their lines:
// "<beat> <data>", the beat's index in the channel (its byte address / 32, at least BEATS) in hex,
// its 32 bytes as one 64-digit hex number, byte j (the channel's byte 32 x beat + j) its bits
// 8 j + 7 .. 8 j (Python: int.from_bytes(b, "little")); the out file's in ascending beat order.
//
// BHW (otpu_mig_pair's bank-half window; 0: none): the address the pair gives is the one the MIG
// decodes (its bank, row and column), and the window's swap again (its own inverse) gives the
// channel's byte the images hold.
//
// BEATS is 1 .. a channel's beats (2^(DAW - 5 - log2 NC)), else a fatal at time 0 (the arrays
// declared at 1 beat at least, so that a bad size is that error and not Verilator's): tb_bcuwg's
// WORDS is an int, and a dense image of 2^31 words a slice or more wraps (bcu1525-s2's 16 GiB:
// WORDS 2^32 = 0, BEATS 0); SimTransport keeps the dense image below that.
module otpu_dimm_model #(
  parameter int BEATS = 1 << 16,          // a channel's 32-byte beats in the dense array (its image)
  parameter int DAW   = 30,               // the DIMM's byte address bits (otpu_mig_pair's CL + log2 NC)
  parameter int DIMM  = 0,
  parameter int NC    = 2,                // its channels (otpu_mig_pair NC),
  parameter int CH0   = NC * DIMM,        //   board channels CH0 .. CH0 + NC - 1
  parameter int IDW   = 4,
  parameter int ARQ   = 64,
  parameter int AWQ   = 32,
  parameter int BHW   = 0,                // otpu_mig_pair's bank-half window (a channel's bytes, log2)
  parameter int BHB   = 16                // and the MIG map's bank-high bit
) (
  input  logic           clk,
  input  logic           rst,
  input  logic           dump,
  input  logic           awvalid,
  output logic           awready,
  input  logic [IDW-1:0] awid,
  input  logic [DAW-1:0] awaddr,
  input  logic [7:0]     awlen,
  input  logic           wvalid,
  output logic           wready,
  input  logic [511:0]   wdata,
  input  logic [63:0]    wstrb,
  input  logic           wlast,
  output logic           bvalid,
  input  logic           bready,
  output logic [IDW-1:0] bid,
  output logic [1:0]     bresp,
  input  logic           arvalid,
  output logic           arready,
  input  logic [IDW-1:0] arid,
  input  logic [DAW-1:0] araddr,
  input  logic [7:0]     arlen,
  output logic           rvalid,
  input  logic           rready,
  output logic [IDW-1:0] rid,
  output logic [511:0]   rdata,
  output logic [1:0]     rresp,
  output logic           rlast,
  output longint         nrd,             // R beats given, W beats taken
  output longint         nwr,
  output int             nerr
);
  typedef struct { logic [DAW-1:0] a; int n; logic [IDW-1:0] id; longint ts, tr; bit got;
                   logic [15:0][511:0] d; int seq; logic [15:0] bk, bm; longint bt [16]; } rd_t;
  typedef struct { logic [DAW-1:0] a; int n; logic [IDW-1:0] id; int seq; } aw_t;
  typedef struct { logic [DAW-1:0] a; int n; logic [IDW-1:0] id; longint ta, tb; bit done;
                   logic [15:0][511:0] d; logic [15:0][63:0] s; int seq; } wr_t;
  localparam int LNC = $clog2(NC);
  localparam longint CHB = longint'(1) << (DAW - 5 - LNC);   // a channel's beats
  localparam int DB = BEATS < 1 ? 1 : BEATS;                 // (the arrays' size: BEATS checked)
  logic [255:0] mem [NC][DB];              // a channel's beats below BEATS
  logic [255:0] hi [NC][longint];          // and those at or above it, loaded or written
  rd_t    rq [$];
  aw_t    awq [$];
  wr_t    wq [$];
  wr_t    wcur;
  int     wk = 0;                          // W beats of the head AW taken
  int     rk = 0;                          // R beats of the current read given
  int     ri = -1;                         // the read whose beats R gives (-1: none)
  int     lat = 75, wlat = 40, jit = 0, bw = 95, st = 0, ooo = 0;
  longint cyc = 0, tb_last = 0, ta_last = 0;
  int     tok = 0;
  logic   p_arv = 1'b0, p_awv = 1'b0, p_wv = 1'b0, p_arr = 1'b0, p_awr = 1'b0, p_wr = 1'b0;
  logic [DAW-1:0] p_ara, p_awa;
  logic [7:0]   p_arl, p_awl;
  logic [511:0] p_wd;
  logic [63:0]  p_ws;
  logic         p_wl;

  // ---- +dimm_mc=1: the MIG (DDR4 clocks, tCK; a clk cycle is four)
  localparam int T_RCD = 17, T_RP = 17, T_RAS = 39, T_RTP = 10, T_WR = 20, T_CWL = 12,
                 T_RRD_S = 4, T_RRD_L = 6, T_FAW = 26, T_WTR_S = 4, T_WTR_L = 10, T_RTW = 11,
                 T_CCD_L = 6, T_REFI = 9363, T_RFC = 421;
  typedef struct { bit we; int seq; int k; logic [DAW-1:0] a; int g; int b; int row;
                   logic [511:0] d; logic [63:0] s; } tx_t;  // a 64-byte command
  typedef struct { int seq; logic [DAW-1:0] a; int n; longint t; int ord; } fb_t;  // a burst at the front end
  int     mc = 0, mc_rl = 32, mc_al = 14, mc_bl = 8, mc_bg = 6, mc_ba = 8, mc_bg1 = -1, mc_ba1 = -1;
  int     mc_rw = 32, mc_rs = 0, mc_strict = 0, mc_sc = 0, mc_scq = 32;
  int     mc_xrtw = 0, mc_xwtr = 0, mc_xbsw = 0, mc_xrcd = 0, mc_xbq = 0;   // calibration (card-gap)
  int     mc_wnl = 0, mc_wrate = 1;
  int     mc_wbg = -1, mc_wbg1 = -1, mc_wba = -1, mc_wba1 = -1;            // the window's map (card-gap)
  int     mc_bgx = -1;                                                     // XOR into the group's high bit
  int     mc_xccd = 0;                                                     // calibration: tCCD_L + N
  int     mc_xgq = 0;                                                      // calibration: a group's CAS FIFO fill rate
  int     mc_xccdb = 0;                                                    // +mc_xccd inside a burst only
  int     gseq [4];                        // a group's last CAS's burst
  int     gch [4];                         // and its channel
  longint gpush [4];                       // a group's last move into its CAS FIFO
  int     gcb [4];                        // a group's last CAS's bank (+mc_xbsw)
  int     rwin = 0, mi = 0, mk = 0;        // reads in the window; the next beat to leave the MIG
  int     scq = 0;                         // beats out of the MIG's port, not yet given on R
  int     sord [$];                        // STRICT: the groups of the commands taken, in order
  tx_t    gq [4][$];                       // a group's transaction FIFO
  tx_t    gc [4][$];                       // its CAS FIFO
  int     gst [4];                         // 0 accept, 1 precharge wait, 2 tRP, 3 activate wait, 4 CAS wait
  bit     gpre [4], gact [4];
  bit     bo [4][4];                       // [group][bank] open, its row, its last ACT / PRE / RD / WR
  int     brow [4][4];
  longint bact [4][4], bpre [4][4], brd [4][4], bwr [4][4];
  longint gcas [4], gacl [4];              // a group's last CAS, last ACT
  longint lrd = -1000, lwr = -1000, lact = -1000;
  longint fawq [4];                        // the last four ACTs
  int     lwr_g = 0, cas_rr = 0, pre_rr = 0, act_rr = 0;
  bit     lwe = 1'b0;                      // the last CAS a write
  fb_t    far [$], faw [$];                // the front end's read and write bursts
  int     fk = 0;                          // the head burst's commands given (the direction's)
  bit     rnw_i = 1'b1, rd_hold = 1'b0, wr_hold = 1'b0;
  int     rd_wl = 0, wr_wl = 0, rd_sc = 2, wr_sc = 2;
  int     tags = 0, wbuf = 0, seqn = 0, aw_ord = 0, w_ord = 0;
  longint ref_due = T_REFI, ref_end = 0;
  bit     ref_blk = 1'b0;
  longint n_act = 0, n_rd = 0, n_wr = 0, n_ref = 0, n_turn = 0;
  int     afd = 0;                         // +mc_actlog's file
  byte    acl [4][4];                      // why the bank is closed: "C", "P", "R"
  bit     alw [4][4];                      // its last CAS a write, at that address
  logic [DAW-1:0] ala [4][4];

  // a DIMM byte address's group, bank and row (ROW_COLUMN_BANK, each field bit where mc_bg* /
  // mc_ba* put it, or in the bank-half window mc_wbg* / mc_wba*: an address map in front of the MIG's)
  function automatic bit fld(input int i, input bit w);
    if (w) return i == mc_wbg || i == mc_wbg1 || i == mc_wba || i == mc_wba1;
    return i == mc_bg || i == mc_bg1 || i == mc_ba || i == mc_ba1;
  endfunction
  function automatic void mdec(input logic [DAW-1:0] a, output int g, output int b, output int row);
    int col, n;
    bit w;
    w = mc_wbg >= 0 && BHW != 0 && (a >> (BHW + LNC)) == '0;
    if (w) begin g = int'({a[mc_wbg1], a[mc_wbg]}); b = int'({a[mc_wba1], a[mc_wba]}); end
    else begin
      g = int'({a[mc_bg1] ^ (mc_bgx >= 0 ? a[mc_bgx] : 1'b0), a[mc_bg]});
      b = int'({a[mc_ba1], a[mc_ba]});
    end
    row = 0; col = 0; n = 0;
    for (int i = 6; i < DAW; i++)
      if (!fld(i, w)) begin
        if (col < 7) col++;
        else begin row = row | (int'(a[i]) << n); n++; end
      end
  endfunction
  // a CAS's timing: tRCD from its ACT, tCCD_L from its group's last, the turnarounds
  function automatic bit cas_ok(input tx_t h, input longint now);
    if (now < bact[h.g][h.b] + T_RCD + mc_xrcd ||
        now < gcas[h.g] + T_CCD_L + (mc_xccdb == 0 || (mc_xccdb == 1 && gseq[h.g] == h.seq) ||
                                     (mc_xccdb == 2 && gch[h.g] == ch_of(h.a)) ? mc_xccd : 0)) return 1'b0;
    if (mc_xbsw != 0 && gcb[h.g] != h.b && now < gcas[h.g] + mc_xbsw) return 1'b0;
    if (h.we) return now >= lrd + T_RTW + mc_xrtw && now >= lwr + 4 * mc_wrate;
    return now >= lwr + T_CWL + 4 + (lwr_g == h.g ? T_WTR_L : T_WTR_S) + mc_xwtr;
  endfunction

  // a 512-bit DIMM beat's two 32-byte words: channel, word (of the address before the window's
  // swap: otpu_mig_pair's dimm)
  localparam int WB = BHW != 0 ? BHW : 14, HB = BHW != 0 ? BHB : 13;    // (BHW 0: indices in range)
  function automatic logic [DAW-1:0] unbhw(input logic [DAW-1:0] a);
    logic [DAW-1:0] d;
    d = a;
    if (BHW != 0 && a[DAW-1:WB+LNC] == '0) begin
      d[HB] = a[WB-1+LNC];
      d[WB-1+LNC] = a[HB];
    end
    return d;
  endfunction
  function automatic int ch_of(input logic [DAW-1:0] a);
    return int'(a[12 +: LNC]);
  endfunction
  function automatic longint coff(input logic [DAW-1:0] a);   // its channel byte (+mc_actlog)
    logic [DAW-1:0] u;
    u = unbhw(a);
    return longint'({u[DAW-1:12+LNC], u[11:0]});
  endfunction
  function automatic longint word_of(input logic [DAW-1:0] a);
    logic [DAW-1:0] d;
    d = unbhw(a);
    return longint'({d[DAW-1:12+LNC], d[11:5]});
  endfunction
  // channel s's 32-byte word w, read and written (with a byte strobe), dense or sparse
  function automatic logic [255:0] getw(input int s, input longint w);
    if (w < longint'(BEATS)) return mem[s][int'(w)];
    return hi[s].exists(w) ? hi[s][w] : '0;
  endfunction
  task automatic putw(input int s, input longint w, input logic [255:0] d, input logic [31:0] m);
    logic [255:0] v;
    v = getw(s, w);
    for (int j = 0; j < 32; j++) if (m[j]) v[8 * j +: 8] = d[8 * j +: 8];
    if (w < longint'(BEATS)) mem[s][int'(w)] = v;
    else hi[s][w] = v;
  endtask

  initial begin
    string dir;
    if (BEATS < 1 || longint'(BEATS) > CHB)
      $fatal(1, "otpu_dimm_model: BEATS %0d: the dense image is 1 .. %0d beats (a channel's)",
             BEATS, CHB);
    nerr = 0;
    void'($value$plusargs("dimm_lat=%d", lat));
    void'($value$plusargs("dimm_wlat=%d", wlat));
    void'($value$plusargs("dimm_jit=%d", jit));
    void'($value$plusargs("dimm_bw=%d", bw));
    void'($value$plusargs("dimm_st=%d", st));
    void'($value$plusargs("dimm_ooo=%d", ooo));
    void'($value$plusargs("dimm_mc=%d", mc));
    void'($value$plusargs("mc_rl=%d", mc_rl));
    void'($value$plusargs("mc_al=%d", mc_al));
    void'($value$plusargs("mc_bl=%d", mc_bl));
    void'($value$plusargs("mc_rw=%d", mc_rw));
    void'($value$plusargs("mc_rs=%d", mc_rs));
    void'($value$plusargs("mc_strict=%d", mc_strict));
    void'($value$plusargs("mc_sc=%d", mc_sc));
    void'($value$plusargs("mc_scq=%d", mc_scq));
    void'($value$plusargs("mc_xrtw=%d", mc_xrtw));
    void'($value$plusargs("mc_xwtr=%d", mc_xwtr));
    void'($value$plusargs("mc_xbsw=%d", mc_xbsw));
    void'($value$plusargs("mc_xrcd=%d", mc_xrcd));
    void'($value$plusargs("mc_xbq=%d", mc_xbq));
    void'($value$plusargs("mc_xccd=%d", mc_xccd));
    void'($value$plusargs("mc_xgq=%d", mc_xgq));
    void'($value$plusargs("mc_xccdb=%d", mc_xccdb));
    void'($value$plusargs("mc_wnl=%d", mc_wnl));
    void'($value$plusargs("mc_wrate=%d", mc_wrate));
    void'($value$plusargs("mc_bg=%d", mc_bg));
    void'($value$plusargs("mc_ba=%d", mc_ba));
    void'($value$plusargs("mc_bg1=%d", mc_bg1));
    void'($value$plusargs("mc_ba1=%d", mc_ba1));
    if (mc_bg1 < 0) mc_bg1 = mc_bg + 1;
    if (mc_ba1 < 0) mc_ba1 = mc_ba + 1;
    void'($value$plusargs("mc_wbg=%d", mc_wbg));
    void'($value$plusargs("mc_wbg1=%d", mc_wbg1));
    void'($value$plusargs("mc_wba=%d", mc_wba));
    void'($value$plusargs("mc_wba1=%d", mc_wba1));
    if (mc_wbg1 < 0) mc_wbg1 = mc_wbg + 1;
    if (mc_wba < 0) mc_wba = mc_ba;
    if (mc_wba1 < 0) mc_wba1 = mc_wba + 1;
    void'($value$plusargs("mc_bgx=%d", mc_bgx));
    if (mc_bgx == mc_bg1 || mc_bgx >= DAW) $fatal(1, "otpu_dimm_model: +mc_bgx=%0d", mc_bgx);
    begin
      string apre;
      if (mc != 0 && $value$plusargs("mc_actlog=%s", apre)) begin
        afd = $fopen($sformatf("%sd%0d.txt", apre, DIMM), "a");
        if (afd != 0)
          $fwrite(afd, "# dimm %0d NC %0d DAW %0d BHW %0d bg %0d %0d ba %0d %0d\n", DIMM, NC, DAW, BHW,
                  mc_bg, mc_bg1, mc_ba, mc_ba1);
      end
    end
    for (int g = 0; g < 4; g++) begin
      gst[g] = 0; gpre[g] = 1'b0; gact[g] = 1'b0; gcas[g] = -1000; gacl[g] = -1000; gcb[g] = 0;
      gpush[g] = -1000; gseq[g] = -1; gch[g] = -1;
      fawq[g] = -1000;
      for (int b = 0; b < 4; b++) begin
        bo[g][b] = 1'b0; brow[g][b] = 0; bact[g][b] = -1000; bpre[g][b] = -1000;
        brd[g][b] = -1000; bwr[g][b] = -1000;
        acl[g][b] = "C"; alw[g][b] = 1'b0; ala[g][b] = '0;
      end
    end
    for (int s = 0; s < NC; s++)
      for (int i = 0; i < BEATS; i++) mem[s][i] = '0;
    if ($value$plusargs("dir=%s", dir))
      for (int s = 0; s < NC; s++) begin
        int fd, nread;
        logic [31:0] w [DB * 8];
        for (int i = 0; i < BEATS * 8; i++) w[i] = '0;
        fd = $fopen($sformatf("%s/ch%0d.bin", dir, CH0 + s), "rb");
        if (fd != 0) begin
          nread = $fread(w, fd);
          $fclose(fd);
        end
        for (int i = 0; i < BEATS; i++)
          for (int k = 0; k < 8; k++) mem[s][i][32 * k +: 32] = w[8 * i + k];
        fd = $fopen($sformatf("%s/ch%0d_hi.txt", dir, CH0 + s), "r");
        if (fd != 0) begin
          longint b;
          logic [255:0] d;
          while ($fscanf(fd, "%h %h\n", b, d) == 2) begin
            if (b < longint'(BEATS) || b >= (longint'(1) << (DAW - 5 - LNC))) begin
              $display("ERROR dimm %0d: ch%0d_hi.txt beat %h outside %h .. %h", DIMM, CH0 + s,
                       b, BEATS, (longint'(1) << (DAW - 5 - LNC)) - 1);
              nerr++;
            end else hi[s][b] = d;
          end
          $fclose(fd);
        end
      end
    nrd = 0; nwr = 0;
    arready = 1'b0; awready = 1'b0; wready = 1'b0; rvalid = 1'b0; bvalid = 1'b0;
    rresp = 2'b00; bresp = 2'b00; rlast = 1'b0; rdata = '0; rid = '0; bid = '0;
  end

  function automatic int rnd(input int n);   // 0 .. n
    return n > 0 ? int'($urandom % (n + 1)) : 0;
  endfunction
  function automatic bit bad(input logic [DAW-1:0] a, input logic [7:0] l);
    longint n;
    n = 64 * (longint'(l) + 1);
    return a[5:0] != 6'd0 || l > 8'd15 || (longint'(a[11:0]) + n > 4096);
  endfunction

  always @(posedge clk) begin
    cyc++;
    if (rst) begin
      rq.delete(); awq.delete(); wq.delete(); wk = 0; rk = 0; ri = -1;
      far.delete(); faw.delete(); fk = 0; tags = 0; wbuf = 0; rwin = 0; mi = 0; mk = 0; scq = 0; sord.delete();
      for (int g = 0; g < 4; g++) begin gq[g].delete(); gc[g].delete(); gst[g] = 0; gpre[g] = 1'b0; gact[g] = 1'b0; end
      arready <= 1'b0; awready <= 1'b0; wready <= 1'b0; rvalid <= 1'b0; bvalid <= 1'b0;
      p_arv = 1'b0; p_awv = 1'b0; p_wv = 1'b0;
    end else begin
      // ---- the master's side of the handshakes: held valids keep their payload
      if (p_arv && !p_arr && (!arvalid || araddr != p_ara || arlen != p_arl)) begin
        $display("ERROR dimm %0d: AR dropped or changed before ARREADY", DIMM); nerr++;
      end
      if (p_awv && !p_awr && (!awvalid || awaddr != p_awa || awlen != p_awl)) begin
        $display("ERROR dimm %0d: AW dropped or changed before AWREADY", DIMM); nerr++;
      end
      if (p_wv && !p_wr && (!wvalid || wdata != p_wd || wstrb != p_ws || wlast != p_wl)) begin
        $display("ERROR dimm %0d: W dropped or changed before WREADY", DIMM); nerr++;
      end
      p_arv = arvalid; p_ara = araddr; p_arl = arlen; p_arr = arready;
      p_awv = awvalid; p_awa = awaddr; p_awl = awlen; p_awr = awready;
      p_wv = wvalid; p_wd = wdata; p_ws = wstrb; p_wl = wlast; p_wr = wready;

      if (mc != 0) begin
        longint now;
        bit     arv, awv, awd, rnw, want, rdy, en, last, sc_gap;
        tx_t    c;
        fb_t    f;
        now = 4 * cyc;
        // ---- AR, AW: to the front end mc_al cycles on; W: a burst's beats after its AW
        if (arvalid && arready) begin
          rd_t r;
          if (bad(araddr, arlen)) begin
            if (nerr < 10) $display("ERROR dimm %0d: AR address %h len %0d", DIMM, araddr, arlen);
            nerr++;
          end
          r.a = araddr; r.n = int'(arlen) + 1; r.id = arid; r.got = 1'b0; r.seq = seqn++;
          r.bk = '0; r.bm = '0;
          rq.push_back(r);
          f.seq = r.seq; f.a = araddr; f.n = r.n; f.t = cyc + mc_al; f.ord = 0;
          far.push_back(f);
        end
        if (awvalid && awready) begin
          aw_t w;
          if (bad(awaddr, awlen)) begin
            if (nerr < 10) $display("ERROR dimm %0d: AW address %h len %0d", DIMM, awaddr, awlen);
            nerr++;
          end
          w.a = awaddr; w.n = int'(awlen) + 1; w.id = awid; w.seq = seqn++;
          awq.push_back(w);
          f.seq = w.seq; f.a = awaddr; f.n = w.n; f.t = cyc + mc_al; f.ord = aw_ord++;
          faw.push_back(f);
        end
        if (wvalid && wready) begin
          if (awq.size() == 0) begin $display("ERROR dimm %0d: W before its AW", DIMM); nerr++; end
          else begin
            if (wk < 16) begin wcur.d[wk] = wdata; wcur.s[wk] = wstrb; end
            wk++; nwr++; wbuf++;
            if (wlast != (wk == awq[0].n)) begin
              $display("ERROR dimm %0d: WLAST %b at beat %0d of %0d", DIMM, wlast, wk, awq[0].n); nerr++;
            end
            if (wlast || wk == awq[0].n) begin
              wcur.a = awq[0].a; wcur.n = awq[0].n; wcur.id = awq[0].id; wcur.seq = awq[0].seq;
              wcur.tb = -1; wcur.done = 1'b0;
              wq.push_back(wcur);
              void'(awq.pop_front()); wk = 0; w_ord++;
            end
          end
        end

        // ---- refresh: the user interface held, the groups drained, every bank closed
        if (!ref_blk && now >= ref_due) ref_blk = 1'b1;
        if (ref_blk && ref_end == 0) begin
          bit idle;
          idle = 1'b1;
          for (int g = 0; g < 4; g++) begin
            if (gq[g].size() != 0 || gc[g].size() != 0 || gst[g] != 0) idle = 1'b0;
            for (int b = 0; b < 4; b++)
              if (bo[g][b] && (now < bact[g][b] + T_RAS || now < brd[g][b] + T_RTP ||
                               now < bwr[g][b] + T_CWL + 4 + T_WR)) idle = 1'b0;
          end
          if (idle) begin
            for (int g = 0; g < 4; g++)
              for (int b = 0; b < 4; b++) begin
                if (bo[g][b]) acl[g][b] = "R";
                bo[g][b] = 1'b0;
              end
            ref_end = now + T_RP + T_RFC; n_ref++;
          end
        end
        if (ref_end != 0 && now >= ref_end) begin
          ref_end = 0; ref_blk = 1'b0; ref_due += T_REFI;
        end

        // ---- one CAS (the last one's direction first), one PRE, one ACT
        begin
          int pick;
          pick = -1;
          if (mc_strict != 0) begin
            if (sord.size() != 0 && gc[sord[0]].size() != 0 && cas_ok(gc[sord[0]][0], now)) pick = sord[0];
          end else
            for (int pass = 0; pass < 2 && pick < 0; pass++)
              for (int i = 0; i < 4 && pick < 0; i++) begin
                int g;
                g = (cas_rr + i) % 4;
                if (gc[g].size() != 0)
                  if ((pass == 1 || gc[g][0].we == lwe) && cas_ok(gc[g][0], now)) pick = g;
              end
          if (pick >= 0) begin
            c = gc[pick].pop_front();
            if (mc_strict != 0) void'(sord.pop_front());
            if (c.we) begin
              longint w;
              if (!lwe) n_turn++;
              // the beat's two 32-byte words into the memory, under its strobes
              w = word_of(c.a);
              for (int h = 0; h < 2; h++) putw(ch_of(c.a), w + h, c.d[256 * h +: 256], c.s[32 * h +: 32]);
              bwr[c.g][c.b] = now; lwr = now; lwr_g = c.g; wbuf--; n_wr++;
            end else begin
              if (lwe) n_turn++;
              for (int i = 0; i < rq.size(); i++)
                if (rq[i].seq == c.seq) begin
                  longint w;
                  w = word_of(c.a);
                  rq[i].d[c.k] = {getw(ch_of(c.a), w + 1), getw(ch_of(c.a), w)};
                  rq[i].bk[c.k] = 1'b1; rq[i].bt[c.k] = cyc + mc_rl;
                end
              brd[c.g][c.b] = now; lrd = now; n_rd++;
            end
            gcas[c.g] = now; gcb[c.g] = c.b; gseq[c.g] = c.seq; gch[c.g] = ch_of(c.a); lwe = c.we; tags--;
            ala[c.g][c.b] = c.a; alw[c.g][c.b] = c.we;
            cas_rr = (pick + 1) % 4;
          end
        end
        for (int i = 0; i < 4; i++) begin
          int g, b;
          g = (pre_rr + i) % 4;
          if (gpre[g]) begin
            b = gq[g][0].b;
            if (now >= bact[g][b] + T_RAS && now >= brd[g][b] + T_RTP &&
                now >= bwr[g][b] + T_CWL + 4 + T_WR) begin
              bo[g][b] = 1'b0; bpre[g][b] = now; gpre[g] = 1'b0; gst[g] = 2; acl[g][b] = "P";
              pre_rr = (g + 1) % 4;
              break;
            end
          end
        end
        if (ref_end == 0)
          for (int i = 0; i < 4; i++) begin
            int g, b;
            g = (act_rr + i) % 4;
            if (gact[g]) begin
              b = gq[g][0].b;
              if (now >= bpre[g][b] + T_RP && now >= lact + T_RRD_S && now >= gacl[g] + T_RRD_L &&
                  now >= fawq[0] + T_FAW) begin
                if (afd != 0)
                  $fwrite(afd, "%0d %0d %0d %0d %0d %c %0d %0d %0h %0d %0d %0h\n", cyc, g, b, gq[g][0].row,
                          brow[g][b], acl[g][b], gq[g][0].we, ch_of(gq[g][0].a), coff(gq[g][0].a),
                          alw[g][b], ch_of(ala[g][b]), coff(ala[g][b]));
                bo[g][b] = 1'b1; brow[g][b] = gq[g][0].row; bact[g][b] = now;
                lact = now; gacl[g] = now; n_act++;
                fawq[0] = fawq[1]; fawq[1] = fawq[2]; fawq[2] = fawq[3]; fawq[3] = now;
                gact[g] = 1'b0;
                if (gc[g].size() < 2 && now >= gpush[g] + mc_xgq) begin
                  gc[g].push_back(gq[g].pop_front()); gst[g] = 0; gpush[g] = now;
                end else gst[g] = 4;
                act_rr = (g + 1) % 4;
                break;
              end
            end
          end

        // ---- the groups: the head in order (a hit to the CAS FIFO, a miss once its CASes are out)
        for (int g = 0; g < 4; g++)
          case (gst[g])
            0: if (gq[g].size() != 0) begin
                 int b;
                 b = gq[g][0].b;
                 if (bo[g][b] && brow[g][b] == gq[g][0].row) begin
                   if (gc[g].size() < 2 && !(mc_xbq != 0 && gc[g].size() != 0 && gc[g][gc[g].size() - 1].b != b) &&
                       now >= gpush[g] + mc_xgq) begin
                     gc[g].push_back(gq[g].pop_front()); gpush[g] = now;
                   end
                 end else if (gc[g].size() == 0) begin
                   if (bo[g][b]) begin gpre[g] = 1'b1; gst[g] = 1; end
                   else begin gact[g] = 1'b1; gst[g] = 3; end
                 end
               end
            2: begin gact[g] = 1'b1; gst[g] = 3; end
            4: if (gc[g].size() < 2 && now >= gpush[g] + mc_xgq) begin
                 gc[g].push_back(gq[g].pop_front()); gst[g] = 0; gpush[g] = now;
               end
            default: ;
          endcase

        // ---- the front end (RD_PRI_REG) and the user interface: one command a cycle, in order
        arv = far.size() != 0 && far[0].t <= cyc;
        awv = faw.size() != 0 && faw[0].t <= cyc;
        awd = awv && faw[0].ord < w_ord;
        rnw = rnw_i;
        if (rnw_i && !(rd_hold && arv) && awv) rnw = 1'b0;
        else if (!rnw_i && !(wr_hold && awv) && arv) rnw = 1'b1;
        want = rnw ? arv : awd;
        rdy = !ref_blk;
        en = 1'b0; last = 1'b0;
        if (want) begin
          f = rnw ? far[0] : faw[0];
          mdec(f.a + DAW'(64 * fk), c.g, c.b, c.row);
          rdy = !ref_blk && gq[c.g].size() < 2 && tags < 32 && (!rnw || mc_rw == 0 || rwin < mc_rw);
          if (rdy) begin
            en = 1'b1;
            c.we = !rnw; c.seq = f.seq; c.k = fk; c.a = f.a + DAW'(64 * fk); c.d = '0; c.s = '0;
            if (c.we)
              for (int i = 0; i < wq.size(); i++)
                if (wq[i].seq == f.seq) begin
                  c.d = wq[i].d[fk]; c.s = wq[i].s[fk];
                  if (fk == f.n - 1) wq[i].tb = cyc + mc_bl;
                end
            gq[c.g].push_back(c); tags++;
            if (rnw && mc_rw != 0) rwin++;
            if (mc_strict != 0) sord.push_back(c.g);
            fk++;
            last = fk == f.n;
            if (last) begin
              fk = 0;
              if (rnw) void'(far.pop_front()); else void'(faw.pop_front());
            end
          end
        end
        if (!rnw || !rd_hold) begin rd_wl = 0; rd_sc = 2; end
        else if (rdy) begin
          if (!arv) rd_wl = 0; else if (!en) rd_wl++;
          if (en && rd_sc < 256) rd_sc++;
        end
        if (rnw || !wr_hold) begin wr_wl = 0; wr_sc = 2; end
        else if (rdy) begin
          if (!awv) wr_wl = 0; else if (!en) wr_wl++;
          if (en && wr_sc < 256) wr_sc++;
        end
        rd_hold = (rnw || rd_hold) && !(en && rnw && last && rd_sc >= 256) && rd_wl < 16;
        wr_hold = (!rnw || wr_hold) && !(en && !rnw && last && wr_sc >= 256 && mc_wnl == 0) && wr_wl < 16;
        rnw_i = rnw;

        // ---- +mc_rw: the MIG's AXI R port, in AR order, a beat a cycle mc_rl - mc_rs cycles after
        // its CAS, at R mc_rs cycles on
        if (mc_rw != 0 && mi < rq.size() && (mc_sc == 0 || scq < mc_scq))
          if (rq[mi].bk[mk] && rq[mi].bt[mk] - mc_rs <= cyc) begin
            rq[mi].bm[mk] = 1'b1; rq[mi].bt[mk] = cyc + mc_rs; rwin--; scq++;
            mk++;
            if (mk == rq[mi].n) begin mk = 0; mi++; end
          end
        // ---- R: in AR order, a beat a cycle once its CAS is mc_rl cycles old (+mc_rw: once it
        // left the MIG mc_rs cycles ago; +mc_sc: not in the cycle after an RLAST)
        sc_gap = 1'b0;
        if (rvalid && rready) begin
          nrd++;
          rk++;
          if (mc_rw != 0) scq--;
          sc_gap = mc_sc != 0 && rlast;
          if (rk == rq[0].n) begin void'(rq.pop_front()); rk = 0; if (mc_rw != 0) mi--; end
        end
        if (!rvalid || rready) begin
          if (!sc_gap && rq.size() != 0 && rq[0].bk[rk] && (mc_rw == 0 || rq[0].bm[rk]) && rq[0].bt[rk] <= cyc) begin
            rvalid <= 1'b1; rdata <= rq[0].d[rk]; rlast <= rk == rq[0].n - 1; rresp <= 2'b00;
            rid <= rq[0].id;
          end else rvalid <= 1'b0;
        end
        // ---- B: in AW order, mc_bl cycles after its burst's last command went in
        if (bvalid && bready) void'(wq.pop_front());
        if (!bvalid || bready) begin
          if (wq.size() != 0 && wq[0].tb >= 0 && wq[0].tb <= cyc) begin
            bvalid <= 1'b1; bresp <= 2'b00; bid <= wq[0].id;
          end else bvalid <= 1'b0;
        end
        arready <= rq.size() < ARQ && rnd(99) >= st;
        awready <= awq.size() + wq.size() < AWQ && rnd(99) >= st;
        wready  <= awq.size() != 0 && wbuf < 48 && rnd(99) >= st;
      end else begin
        tok += bw;
        // ---- AR: a read takes its data at ts, gives its first beat at tr or later
        if (arvalid && arready) begin
          rd_t r;
          if (bad(araddr, arlen)) begin
            if (nerr < 10) $display("ERROR dimm %0d: AR address %h len %0d", DIMM, araddr, arlen);
            nerr++;
          end
          r.a = araddr; r.n = int'(arlen) + 1; r.id = arid;
          r.tr = cyc + lat + rnd(jit);
          r.ts = cyc + rnd(int'(r.tr - cyc));
          r.got = 1'b0;
          rq.push_back(r);
        end
        foreach (rq[i])
          if (!rq[i].got && rq[i].ts <= cyc) begin
            if (!bad(rq[i].a, 8'(rq[i].n - 1)))
              for (int k = 0; k < rq[i].n; k++) begin
                longint w;
                w = word_of(rq[i].a) + 2 * k;
                rq[i].d[k] = {getw(ch_of(rq[i].a), w + 1), getw(ch_of(rq[i].a), w)};
              end
            rq[i].got = 1'b1;
          end
        // ---- R: one read's beats at a time, a beat a cycle at most (the bus's share)
        if (rvalid && rready) begin
          nrd++;
          rk++;
          if (rk == rq[ri].n) begin rq.delete(ri); rk = 0; ri = -1; end
        end
        if (ri < 0 && rq.size() != 0) begin
          // the oldest read of each ID goes in AR order; another ID's may pass a head not yet due
          if (rq[0].got && rq[0].tr <= cyc) ri = 0;
          else if (ooo > 0 && rnd(99) < ooo)
            for (int i = 1; i < rq.size() && ri < 0; i++) begin
              bit older_same;
              older_same = 1'b0;
              for (int k = 0; k < i; k++) if (rq[k].id == rq[i].id) older_same = 1'b1;
              if (!older_same && rq[i].got && rq[i].tr <= cyc) ri = i;
            end
        end
        if (!rvalid || rready) begin
          if (ri >= 0 && tok >= 100) begin
            tok -= 100;
            rvalid <= 1'b1; rdata <= rq[ri].d[rk]; rlast <= rk == rq[ri].n - 1; rresp <= 2'b00;
            rid <= rq[ri].id;
          end else rvalid <= 1'b0;
        end

        // ---- AW, W: a burst's beats after its AW (the bus's share); once all in, it lands at ta
        // and answers at tb
        if (awvalid && awready) begin
          aw_t w;
          if (bad(awaddr, awlen)) begin
            if (nerr < 10) $display("ERROR dimm %0d: AW address %h len %0d", DIMM, awaddr, awlen);
            nerr++;
          end
          w.a = awaddr; w.n = int'(awlen) + 1; w.id = awid;
          awq.push_back(w);
        end
        if (wvalid && wready) begin
          tok -= 100;
          if (awq.size() == 0) begin $display("ERROR dimm %0d: W before its AW", DIMM); nerr++; end
          else begin
            if (wk < 16) begin wcur.d[wk] = wdata; wcur.s[wk] = wstrb; end
            wk++; nwr++;
            if (wlast != (wk == awq[0].n)) begin
              $display("ERROR dimm %0d: WLAST %b at beat %0d of %0d", DIMM, wlast, wk, awq[0].n); nerr++;
            end
            if (wlast || wk == awq[0].n) begin
              wcur.a = awq[0].a; wcur.n = awq[0].n; wcur.id = awq[0].id;
              wcur.tb = cyc + wlat + rnd(jit);
              if (wcur.tb < tb_last) wcur.tb = tb_last;
              tb_last = wcur.tb;
              wcur.ta = cyc + rnd(int'(wcur.tb - cyc));
              if (wcur.ta < ta_last) wcur.ta = ta_last;
              ta_last = wcur.ta;
              wcur.done = 1'b0;
              wq.push_back(wcur);
              void'(awq.pop_front()); wk = 0;
            end
          end
        end
        foreach (wq[i])
          if (!wq[i].done && wq[i].ta <= cyc) begin
            if (!bad(wq[i].a, 8'(wq[i].n - 1)))
              for (int k = 0; k < wq[i].n; k++)
                for (int h = 0; h < 2; h++)
                  putw(ch_of(wq[i].a), word_of(wq[i].a) + 2 * k + h, wq[i].d[k][256 * h +: 256],
                       wq[i].s[k][32 * h +: 32]);
            wq[i].done = 1'b1;
          end
        // ---- B: in AW order, once landed
        if (bvalid && bready) void'(wq.pop_front());
        if (!bvalid || bready) begin
          if (wq.size() != 0 && wq[0].done && wq[0].tb <= cyc) begin
            bvalid <= 1'b1; bresp <= 2'b00; bid <= wq[0].id;
          end else bvalid <= 1'b0;
        end
        if (tok > 100) tok = 100;

        arready <= rq.size() < ARQ && rnd(99) >= st;
        awready <= awq.size() + wq.size() < AWQ && rnd(99) >= st;
        // W takes the bus's share left after R (R first)
        wready  <= awq.size() != 0 && rnd(99) >= st && tok - (ri >= 0 ? 100 : 0) + bw >= 100;
      end
    end
  end

  always @(posedge clk) if (dump) begin
    string dir;
    for (int g = 0; g < 4; g++)
      for (int q = 0; q < 2; q++)
        for (int i = 0; i < (q == 0 ? gc[g].size() : gq[g].size()); i++) begin
          tx_t c;
          c = q == 0 ? gc[g][i] : gq[g][i];
          if (c.we)
            for (int h = 0; h < 2; h++)
              putw(ch_of(c.a), word_of(c.a) + h, c.d[256 * h +: 256], c.s[32 * h +: 32]);
        end
    if (afd != 0) $fflush(afd);
    if (mc != 0)
      $display("dimm %0d: the MIG model: %0d ACTs, %0d read CASes, %0d write CASes, %0d turnarounds, %0d refreshes over %0d cycles",
               DIMM, n_act, n_rd, n_wr, n_turn, n_ref, cyc);
    if ($value$plusargs("dir=%s", dir)) begin
      if (awq.size() != 0 || wq.size() != 0)
        $display("dimm %0d: %0d write bursts not landed at the dump", DIMM, awq.size() + wq.size());
      for (int s = 0; s < NC; s++) begin
        int fd;
        fd = $fopen($sformatf("%s/ch%0d_out.bin", dir, CH0 + s), "wb");
        for (int i = 0; i < BEATS; i++)
          for (int k = 0; k < 8; k++) $fwrite(fd, "%u", mem[s][i][32 * k +: 32]);
        $fclose(fd);
        fd = $fopen($sformatf("%s/ch%0d_hi_out.txt", dir, CH0 + s), "w");
        foreach (hi[s][b]) $fwrite(fd, "%0h %h\n", b, hi[s][b]);
        $fclose(fd);
      end
    end
  end
endmodule
