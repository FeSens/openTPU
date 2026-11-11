// Slice DRAM ports (as otpu_dram) on NCH memory channels' native ports, CB-byte beats: one command
// per beat. The card's (NCH 2, CB 64, the defaults): the two DDR3 channels, 512-bit data, behind
// otpu_mem_ch and LiteDRAM's native port (docs/litedram.md section 3); the BCU1525 WG targets' (NCH 8, CB 128):
// a slice's 8 channels, each behind a bridge (otpu_hbm_ch) to an AXI port (docs/native_channels.md).
// The slice side, the address map and the behaviour are those of otpu_axi_dram, the MIG AXI
// builds' adapter it replaced. What existed only for AXI and the MIG's AXI front end, whose
// transactions cost a lot, is gone: read and write bursts, their gather timers and 4 KB splits,
// AXI IDs, write responses and err. A native command costs nothing per transaction, so a request
// goes out as soon as it may, one command per beat.
//
// Address map: the slice's byte address space is interleaved over the channels in CB-byte beats:
// logical beat b = addr / CB, wide chunk m = b / NCH (P = NCH x CB bytes: a beat on every
// channel), is channel beat m (n_caddr: CAW bits, a channel's 2^AW / NCH bytes) on channel
// c = (b % NCH) ^ h(m), h = 0 without CHASH and with it the XOR fold of m's bits into log2(NCH)
// bits (bit i of m into bit i % log2(NCH); at NCH 2 the parity of m). A D-byte chunk (port B;
// requests are chunk aligned) is D / CB beats of one wide chunk on as many channels (the card's:
// one beat on each of its two channels; the WG targets': one beat on one channel), and a streamed
// operand uses the channels evenly. CHASH permutes each wide chunk's beats over the channels by
// its index's fold, so the beats at a power-of-two stride of chunks (a transposed V's column: one
// byte per cache row) spread over the channels instead of falling all on one. The host applies
// the same map when it fills and reads the memory (opentpu/host/board.py; CAPS bit7 = CHASH).
//
// The native port, per channel (core clock): a command stream (we, beat); one write-data beat
// (data, byte mask with 1 = write the byte) per write command, in write-command order, which may
// be taken before or after its command; read data in read-command order with no backpressure;
// n_wdone, the count of write beats the controller has taken. One master's commands on one
// channel execute in command order (LiteDRAM: one port, in-order bank queues; the MIG keeps the
// order of accesses to one address): a read issued after a write to its beat returns the written
// data without waiting for n_wdone, and a write counted in n_wdone is visible to the other master
// (XDMA, the host) too.
//
// Per channel, one command a cycle, from (in this priority) the A queue (a read's run of APF
// commands, or a write), the SW queue's next fill read, the SW queue's oldest write, the B queue
// (a read or a write), the W queue (port W's reads). A command shown and not taken is shown again
// unchanged (hv); a write shows its command and its data together, until both are taken (c_done,
// d_done). Reads need response room, reserved as the command goes: a B read one entry of its
// channel's RD-beat FIFO, an A run APF entries of the AD-beat FIFO; an SW fill read's data goes
// to its queue slot, a W read's to the MXU (port W, below: at most XD W reads in flight a
// channel). Each channel's reads return in command order, so a tag FIFO per channel (source, SW
// slot; its head registered) routes every returning beat to B, W, A or the SW slot. The response
// FIFOs check for overflow in simulation: read data never arrives without reserved room.
//
// Port B: chunk reads and word-masked chunk writes; a partial beat goes out with its byte mask
// (the channel module does the controller's read-modify-write, or the MIG's wr_bytes). Port A:
// single-word reads and byte-enabled word writes (a partial beat). Port SW: byte-enabled word
// writes (the quantizer's QST stores), independent of A so that they never hold up the MXU's
// scale reads. SW writes gather per channel in a one-beat buffer: writes to the same beat merge,
// and the beat goes to the SW queue when a write to another beat arrives, when all its bytes are
// written, or after WGATHER cycles without an SW write (so a QST's byte stream costs one write per
// beat, not per byte). wr_idle stays low while a beat is gathered, so a QST still completes only
// once its writes are in memory. Gathered beats go out in order through one queue per channel; a
// beat with bytes missing (a transposed V column: one byte per beat) first reads its beat (a fill
// read) and is written whole, so the controller's read-modify-write, which holds the channel for a
// read's whole latency, never runs for a QST (SW_FILL, the cards' DDR3 controllers). Without
// SW_FILL a partial beat goes out with its byte mask, one command: for a memory that takes masks
// itself (the WG targets' memory: data mask on, no ECC), where the fill read would only cost a read and
// the logic (otpu_hbm_ch passes the mask as WSTRB). The queue holds WQD beats per channel, so
// that the fill reads of a transposed V column overlap (a slot lives from its push until its
// write goes out: about a read latency). The queue's writes wait while its fill reads still go
// out (unless it is half full), so the controller sees runs of reads, then runs of writes.
// A reads that fall in the beat of the previous A read (the MXU's scale stream: CB / 4 scales per
// beat) reuse it without a DRAM access. An A read that misses fetches a run of APF
// channel-consecutive beats (APF read commands); the A reads that follow the run in order take
// its beats without a DRAM access, and a read off the run (or after an A write) drops the beats
// not yet taken.
// Port W (present when P > D: the WG targets' 1 KiB wide chunks): P-byte reads, a wide chunk each (one
// beat on every channel at the same channel beat), for the MXU's weight streams at WG > 1 (weight
// groups: the port plan's I2). The chunk is not assembled here: each channel's beats go out as
// they return, a cycle later (lane c of w_bv / w_bd: channel c's, in its requests' order; the
// lanes independent, no backpressure). Lane c holds logical beat c ^ h(m) of the wide chunk; a
// grouped operand stores its chunks' rows in that order (docs/isa.md "Weight groups"), so lane c
// is row c, the MXU's column c. The MXU writes each lane into the chunk FIFO entry it reserved
// before the request (its credit: the entry is the room), so the chunk is stored once, in the
// MXU's FIFO, and nothing permutes it on the way. Its queue's
// reads go after A's, SW's and B's on each channel: the weight stream is always backlogged, and
// B's demand is bounded (a D-byte chunk a cycle), so B's reads and writes (a DSTEP's state) never
// wait behind it.
//
// Posted write data (WPOST, the WG targets': its bridges' write-data FIFOs, otpu_hbm_ch u_wd, take a
// channel's data ahead of its commands). A B write's beat and a gathered SW beat go to their
// channel's write data when they are taken (a B beat from a stage register the cycle after its
// take, an SW beat from gd the cycle after its push), not when their command goes out, so neither
// the B queue nor the SW queue holds write data (2 x 1 KiB of LUT RAM a channel). The channel
// takes every beat posted (n_wvalid without n_wready): a take needs room for what may follow it
// before the room is seen (ok: n_wfree, the FIFO's free beats, at least PM) on every channel,
// as b_rdy and sw_rdy cannot know the take's channel. One beat a cycle a channel: the SW beat
// first; a B beat that meets one waits a cycle in its stage (bh; B and SW take nothing then). The
// channel writes its FIFO's head beats in its write commands' order, so a channel's writes go out
// in the order their data went in: a write-order FIFO per channel holds each beat's source, and a
// write goes out only from the source at its head, or, with the FIFO empty, from the source whose
// beat goes in this cycle; reads go as before. Needs SW_FILL 0 (a fill read's data is merged as
// the write goes out) and a chunk on one channel (BPB 1); port A carries no writes (otpu_slice:
// the MXU's scale reads).
//
// One gather buffer (SWG, with WPOST; the WG targets' area cut: docs/native_channels.md): the SW writes
// gather in one beat for every channel instead of one a channel (the buffer holds its channel
// too), so an SW write to another beat, on any channel, pushes it; and at most one beat goes in a
// cycle over all the channels: a B beat waits a cycle in its stage when an SW beat goes in on any
// channel, not only on its own. The posted beat is then one bus, the same on every channel
// (n_wdata, n_wmask; n_wvalid the channel's), with no selection per channel and one gather
// beat's flip-flops, not NCH. SW writes that alternate between beats push each one partial (more,
// smaller writes); a stream that fills a beat before the next one merges as before.
//
// Posted W beats (XPOST, the WG targets' (a) v2: docs/native_channels.md): a W read's beats do not come
// back here. Each W request carries the MXU chunk FIFO entry it reserved (w_slot); its channel
// commands carry their source and the slot (n_csrc, n_cslot), and the channel writes each beat
// into lane c of that entry itself (the bridge, otpu_hbm_ch; the co-simulation's memory model),
// counting the beats written (p_cntg, gray-coded on the channel's clock). The counts are
// synchronized and decoded here once (p_cnt: the MXU's entry completion) and return the W reads
// in flight (xi: rd_idle), so a posted beat cannot land after rd_idle; no W read enters the tag
// FIFO, and w_bv / w_bd stay 0. A one-beat W read (w_one: a natural MM's D-byte chunk at w_addr,
// the MXU's at WG > 1 under XPOST, port B's no more) goes into its channel's W queue alone, so
// every posted read of a channel, grouped or not, goes out in the MXU's port-W order, the order
// its per-lane counts complete entries in.
//
// Half beats (RHB, WHB: the WG targets' R1 / R2, docs/native_channels.md; 0, the default: whole beats).
// The paths between the adapter and its channels at half a beat a cycle, their LUT RAMs at half the
// width and twice the entries. RHB: each read beat comes in two halves on n_rvalid, the low one first
// (n_rdata's low CB x 4 bits; a toggle a channel says which half), routed by the same tag (the tag
// FIFO moves on the high half). The B and A FIFOs hold half beats: 2 RD and 2 AD entries of CB x 4
// bits; their room (rb_res, ra_res) is still counted in beats. Port B takes a chunk out over two
// cycles: the oldest read's low half into b_lo, then the chunk (b_rdata: the high half and b_lo)
// with b_rvalid; in that cycle the next read, if it is on another channel and its low half is in,
// takes it into b_lo (b_ov), so reads on different channels go a chunk a cycle, and two on one
// channel one every two cycles (b_lo one register a slice). Port A takes a beat out in one cycle: each channel
// holds its head beat's low half in a register of its own (alo), loaded by the FIFO's read port in
// a cycle it does not read a high half, so a beat goes out (a_rdata: the head's high half and alo)
// as soon as its high half is in, every cycle when the beats come from different channels (the
// scale stream's consecutive beats are on different channels), every other cycle from one. A read
// that drops a run's beats (drop) first skips them in a cycle of its own. Needs XPOST, SW_FILL 0, a
// chunk on one channel and AB = CB: only port A's and B's reads come back on n_rdata. WHB: a posted
// beat goes onto the bus (n_wdata's low CB x 4 bits, n_wmask's low CB / 2; n_wvalid each half) in
// two cycles, the low half as it goes in (as at WHB 0), the high half the next cycle from a
// register (wh_*), and no beat goes in in that cycle: an SW take or push waits for a cycle with no
// beat going in, and the B stage waits for one with the bus free (bh). Needs WPOST and SWG. n_wfree
// stays in beats (the channel's FIFO holds half beats).
//
// Ordering. otpu_axi_dram waited for AXI write responses; here the command stream orders instead.
// Once a command has entered a channel's stream, stream order is memory order. Before that, the
// A, SW, W and B queues may reorder against each other, as AXI's read and write channels did: the
// slice orders what depends across ports (a read that depends on a QST or a B write comes after
// wr_idle; a write over what a read still has to read, after its data: rd_idle). Within a queue,
// requests go out in order. So:
// - An SW fill read waits until every older SW write of its beat has entered the stream (not
//   until it is done): a counter per bucket (a hash of the beat) counts the slots past the read
//   point whose write has not gone out, and a read waits while its bucket's count is not zero (a
//   collision only delays it). The fill read then follows those writes in the stream and reads
//   their data.
// - (SW_FILL) A fill read and its beat's write are not atomic: a B write, or the host's, to the beat's
//   other bytes between them would be lost. The slice keeps B writes out (a QST's scoreboard
//   footprint covers every word of its beats, otpu_pkg beats(), so an ST, DSTEP or STREAM into
//   one waits for the QST or the QST for it); the host must not write the beats of a running
//   QST (docs/isa.md, QST).
// - A write drops channel c's A run, or the reused beat, when it has gone out on c (command and
//   data taken; the stream carries nothing else meanwhile) and its beat is in them: the run's
//   commands before it read the old beat, the ones after it the new. The drop lands at the end
//   of the cycle after the write goes out. An A read taken before then may still take the old
//   beat; it does not depend on the write: the slice waits for wr_idle, which is high in that
//   cycle at the earliest, and an A request then waits a cycle in its port A register. An A
//   write also drops them when it is taken: the A reads after it in the A queue would otherwise
//   take a beat fetched before it. So the QSTs that stream while an MM runs, and DSTEP's state
//   write-back, cost the MXU's scale stream no runs.
// - wr_idle: every accepted write has gone out and is counted in n_wdone.
// - rd_idle: every A, W and B read taken has returned its data (a read still in one of the
//   slice's port registers is the slice's: otpu_slice, otpu_board).
//   This adapter does not reset with the run (the board's RUN), so a run the host stops leaves
//   its reads to come back after it: the loader and the next run wait for rd_idle (otpu_slice,
//   otpu_board). An A run's beats not asked for are not in it (the next miss drops them).
// - The host's writes come through the channels' other master (XDMA), which this adapter does not
//   see, so they drop neither the reused beat nor the runs. a_flush drops both where the host may
//   have written since the beats were fetched: between runs (the core's reset), at a program load
//   and when a WAITW's word holds (the host writes the words a program reads after a WAITW before
//   the word it waits for). Their beats not yet taken are dropped by the next miss, as after a
//   write. Without it a run's first A reads took the previous run's beat over the host's rewrite
//   (the 35B's back-to-back embed runs: token 0's block scales for tokens 1 and 2).
// Requests are taken when req && rdy; rdy depends on registered state only. Reads return in
// order per port (the B tags with their data; W's beats in order per channel).
module otpu_native_dram #(
  parameter int D = 128,
  parameter int NCH = 2,                             // channels (a power of two, >= 2)
  parameter int CB = 64,                             // channel beat bytes (64, 128)
  parameter int QD = 16,                             // request queue depth per channel and port
  parameter int WQD = 32,                            // SW (gathered beat) queue depth per channel
  parameter bit CHASH = 1'b1,                        // address map: see the top
  parameter int WGATHER = 4,                         // idle cycles before a gathered SW beat goes out
  parameter bit SW_FILL = 1'b1,                      // a partial SW beat: read, merged and written
                                                     //   whole (1), or written with its mask (0)
  parameter int RD = 256 / NCH,                      // B read beats in flight per channel (256 a
                                                     //   slice: B's reads, a D chunk a cycle, spread
                                                     //   over the channels; 128 on the cards')
  parameter bit RB_BLOCK = 1'b0,                     // the B FIFO in block RAM at any depth (RBB)
  parameter int AD = 64,                             // A read beats in flight per channel
  parameter int XD = 64,                             // W read beats in flight per channel
  // A read run (prefetch), beats. 32: decode's knee (docs/litedram.md section 11; 8 -> 32 is
  // 0.9-1.3% fewer cycles per token, 64 no better): fewer runs, so fewer row and bank changes
  // between the scale stream and the others
  parameter int APF = 32,
  // port A's read bytes: 32 (the aligned 32 bytes holding the word), or CB (its beat: the MXU's
  // scales at WG > 1)
  parameter int AB = 32,
  parameter bit WPOST = 1'b0,                        // write data posted at the take: see the top
  parameter bit SWG = 1'b0,                          // WPOST: one SW gather buffer and one posted
                                                     //   beat a cycle for every channel (see the top)
  parameter int WFW = 6,                             // n_wfree's bits (otpu_hbm_ch WAW)
  parameter bit XPOST = 1'b0,                        // W beats posted by the channels: see the top
  parameter int XSW = 9,                             // XPOST: a W read's chunk FIFO entry's bits
  parameter int PCW = XSW + 2,                       // XPOST: the posted-beat counts' bits (a
                                                     //   lane's entries in flight, x 2)
  parameter bit RHB = 1'b0,                          // read data in half beats: see the top
  parameter bit WHB = 1'b0,                          // posted write data in half beats: see the top
  parameter int AW = 32,                             // the byte address bits (otpu_seq AW)
  parameter int SWW = 1,                             // the SW port's words: a write's bytes are
                                                     //   the aligned 4 SWW holding word sw_addr
  parameter int WA = (AW > 34) ? AW - 2 : 32         // the ports' word address bits (derived:
                                                     //   otpu_pkg waw)
) (
  input  logic              clk,
  input  logic              rst,
  input  logic              a_flush,    // drop port A's reused beat and runs (see Ordering)
  // slice side
  output logic              a_rdy_x,
  input  logic              a_req_x,
  input  logic              a_we_x,
  input  logic [WA-1:0]     a_addr_x,   // word address
  input  logic [31:0]       a_wdata_x,
  input  logic [3:0]        a_be_x,
  output logic              a_rvalid,
  output logic [AB*8-1:0]   a_rdata,    // the aligned AB bytes holding word a_addr (of its beat)
  output logic              sw_rdy,
  input  logic              sw_req,
  input  logic [WA-1:0]     sw_addr,    // word address
  input  logic [32*SWW-1:0] sw_wdata,
  input  logic [4*SWW-1:0]  sw_be,
  output logic              b_rdy,
  input  logic              b_req,
  input  logic              b_tag,
  input  logic              b_we,
  input  logic [D/4-1:0]    b_wmask,
  input  logic [D*8-1:0]    b_wdata,
  input  logic [WA-1:0]     b_addr,     // word address (chunk aligned)
  input  logic [$clog2(NCH)-1:0] b_par, // the fold of b_addr's wide chunk (the channel hash, CHASH
                                        //   or not: otpu_slice chash; ^b_addr[WA-1:5] on the card)
  output logic              b_rvalid,
  output logic              b_rtag,
  output logic [D*8-1:0]    b_rdata,
  output logic              w_rdy,      // port W (P > D; else never ready, never valid)
  input  logic              w_req,
  input  logic [WA-1:0]     w_addr,     // word address (P aligned)
  input  logic [XSW-1:0]    w_slot,     // XPOST: the MXU chunk FIFO entry it reserved
  input  logic              w_one,      // XPOST: one beat, the D-byte chunk at w_addr (its channel)
  output logic [NCH-1:0]    w_bv,       // channel c's beat of a W read (in its reads' order)
  output logic [NCH*CB*8-1:0] w_bd,     // lane c: channel c's beat
  output logic              wr_idle,
  output logic              rd_idle,    // no A, W or B read in flight (taken, its data not yet out)
  // native memory masters, one per channel ([NCH-1:0] = channel)
  output logic [NCH-1:0]        n_cvalid,
  input  logic [NCH-1:0]        n_cready,
  output logic [NCH-1:0]        n_cwe,
  output logic [NCH-1:0][AW-1-$clog2(NCH*CB):0] n_caddr,  // the beat in the channel (CAW bits)
  output logic [NCH-1:0][1:0]   n_csrc,     // a read's source (K_*: B, A, SW fill, W)
  output logic [NCH-1:0][XSW-1:0] n_cslot,  // XPOST: a W read's chunk FIFO entry
  output logic [NCH-1:0]        n_wvalid,   // write data: one beat per write command, in order
  input  logic [NCH-1:0]        n_wready,
  output logic [NCH-1:0][CB*8-1:0] n_wdata,
  output logic [NCH-1:0][CB-1:0] n_wmask,    // 1 = write the byte
  input  logic [NCH-1:0]        n_rvalid,   // read data, in read-command order, no backpressure
  input  logic [NCH-1:0][CB*8-1:0] n_rdata,
  input  logic [NCH-1:0][15:0]  n_wdone,    // write beats taken by the controller (mod 2^16)
  input  logic [NCH-1:0][WFW-1:0] n_wfree,  // WPOST: the write-data FIFO's free beats (from
                                            //   registers; never more than are free)
  input  logic [NCH-1:0][PCW-1:0] p_cntg,   // XPOST: the W beats channel c has posted into
                                            //   the MXU (mod 2^PCW, gray, the channel's clock)
  output logic [NCH-1:0][PCW-1:0] p_cnt     // XPOST: p_cntg synchronized and decoded
);
  localparam int CL = $clog2(NCH);                   // channel bits (the hash's)
  localparam int PWW = $clog2(NCH * CB / 4);         // word-address bits in a wide chunk
  localparam int CAW = AW - 2 - PWW;                 // channel beat bits (a channel's 2^AW / NCH B)
  initial if (AW < 32 || WA != ((AW > 34) ? AW - 2 : 32))
    $fatal(1, "otpu_native_dram: AW %0d (32 or more), WA %0d (derived)", AW, WA);
  initial if (NCH < 2 || NCH & (NCH - 1) || !(CB == 64 || CB == 128) || D % CB || (NCH * CB) % D)
    $fatal(1, "otpu_native_dram: NCH a power of two >= 2, CB 64 or 128, D whole beats, P whole chunks");
  // (AD >= 2 APF - 1: a run can start while the previous run's untaken beats are stored)
  initial if (APF < 1 || APF > 64 || AD < 2 * APF - 1) $fatal(1, "otpu_native_dram: bad APF");
  initial if (RD < 2 || RD & (RD - 1) || AD & (AD - 1) || WQD < 2 || WQD & (WQD - 1) || XD < 1)
    $fatal(1, "otpu_native_dram: RD, AD and WQD powers of two (their pointers wrap), XD >= 1");
  localparam bit HAS_W = NCH * CB > D;               // port W: a wide chunk is more than a chunk
  localparam int WB = $clog2(CB / 4);                // word-address bits in a beat
  localparam int SB = 4 * SWW;                       // an SW write's bytes (aligned)
  localparam int SSL = $clog2(SWW);
  initial if (AB != 32 && AB != CB) $fatal(1, "otpu_native_dram: AB %0d (32 or CB)", AB);
  localparam int BPB = D / CB;                       // beats of a B chunk
  localparam int BL = $clog2(BPB);
  initial if (XPOST && !HAS_W) $fatal(1, "otpu_native_dram: XPOST needs port W (P > D)");
  initial if (XPOST && BPB != 1) $fatal(1, "otpu_native_dram: XPOST needs a chunk on one channel");
  initial if (WPOST && (SW_FILL || BPB != 1))
    $fatal(1, "otpu_native_dram: WPOST needs SW_FILL 0 and a chunk on one channel (BPB %0d)", BPB);
  initial if (SWG && !WPOST) $fatal(1, "otpu_native_dram: SWG needs WPOST");
  initial if (RHB && (!XPOST || SW_FILL || BPB != 1 || AB != CB))
    $fatal(1, "otpu_native_dram: RHB needs XPOST, SW_FILL 0, a chunk on one channel and AB = CB");
  initial if (WHB && !(WPOST && SWG)) $fatal(1, "otpu_native_dram: WHB needs WPOST and SWG");
  localparam int HDW = RHB ? CB * 4 : CB * 8;        // read data bits a cycle (a half or a beat)
  localparam int WDW = WHB ? CB * 4 : CB * 8;        // posted write data, mask bits a cycle
  localparam int WMW = WHB ? CB / 2 : CB;
  localparam int RE = RHB ? 2 * RD : RD;             // the B and A FIFOs' entries
  localparam int AE = RHB ? 2 * AD : AD;
  localparam int KB = (CL - BL > 0) ? CL - BL : 1;   // a chunk's index in its wide chunk
  localparam int QW = $clog2(QD);
  localparam int WW = $clog2(WQD);
  localparam int NH = 64;                            // SW hazard buckets (see w_blk)
  localparam int HW = $clog2(NH);
  localparam int RW = $clog2(RD);
  localparam int AW_ = $clog2(AD);
  localparam int REW = $clog2(RE);                   // (RHB: a half beat's entry)
  localparam int AEW = $clog2(AE);
  localparam int XW = $clog2(XD);
  localparam int RNW = (APF > 1) ? $clog2(APF) : 1;  // a run's beat index
  localparam int OD = 256;                           // B / W tags, A order entries in flight (2^k)
  localparam int OW = $clog2(OD);
  localparam int TD = 2 ** $clog2(RD + AD + WQD + ((HAS_W && !XPOST) ? XD : 0));  // reads in
                                                     //   flight per channel (tagged)
  localparam int TW = $clog2(TD);
  localparam int PW = $clog2(NCH * (2 * QD + WQD) + 1);  // writes accepted and not gone out
  localparam logic [1:0] K_B = 2'd0, K_A = 2'd1, K_W = 2'd2, K_X = 2'd3;   // a read's source (X: W)
  localparam logic [2:0] S_A = 3'd0, S_WR = 3'd1, S_WW = 3'd2, S_B = 3'd3, S_X = 3'd4;  // a command's

  // the hazard bucket of a channel beat: an XOR fold, so that the strided beats of a transposed
  // V column spread over the buckets
  function automatic logic [HW-1:0] whash(input logic [CAW-1:0] a);
    logic [HW-1:0] h;
    h = '0;
    for (int i = 0; i < CAW; i++) h[i % HW] ^= a[i];
    return h;
  endfunction

  // n + inc - dec as a choice among n - 1, n and n + 1: the counts the slice's requests move
  // (the B and SW queues, the B tags) take their late strobes after the adders, not before
  function automatic logic [15:0] updn(input logic [15:0] n, input logic inc, input logic dec);
    return inc == dec ? n : inc ? n + 1'b1 : n - 1'b1;
  endfunction

  // the fold of a word address's wide chunk index (bits WA-1:PWW) into CL bits, and the channel
  // of its beat (the hash with CHASH)
  function automatic logic [CL-1:0] fold(input logic [WA-1:0] word_addr);
    logic [CL-1:0] h;
    h = '0;
    for (int i = PWW; i < WA; i++) h[(i - PWW) % CL] ^= word_addr[i];
    return h;
  endfunction
  function automatic logic [CL-1:0] chan_of(input logic [WA-1:0] word_addr);
    return word_addr[PWW-1:WB] ^ (CHASH ? fold(word_addr) : CL'(0));
  endfunction

  // ------------------------------------------------------------------ request queues
  // per channel: qb (port B beats), qa (port A beats), qw (gathered SW beats), qx (port W beats);
  // m: the channel beat
  typedef struct packed {
    logic           we;
    logic [CAW-1:0] m;
    logic [CB*8-1:0] data;
    logic [CB/4-1:0] wmask;    // word enables
  } qb_t;
  typedef struct packed {
    logic           we;
    logic [CAW-1:0] m;
    logic [WB-1:0]  idx;       // word in the beat
    logic [31:0]    data;
    logic [3:0]     be;
  } qa_t;
  typedef struct packed {      // a gathered SW beat
    logic [CAW-1:0] m;
    logic [CB*8-1:0] data;
    logic [CB-1:0]  strb;
  } qw_t;
  typedef struct packed {      // a read in flight: its source, and an SW fill read's slot
    logic [1:0]    k;
    logic [WW-1:0] slot;
  } tg_t;

  logic [QW:0] qb_n [NCH];
  logic [QW:0] qb_h [NCH];                  // (2 QD slots: see qbm)
  logic [QW:0] qa_n [NCH];
  logic [QW:0] qa_h [NCH];                  // (2 QD slots: see qbm)
  logic [QW:0] qx_n [NCH];
  logic [QW-1:0] qx_h [NCH];
  logic [RNW-1:0] a_iss [NCH];              // the head A read's run: beats already issued
  // port SW (gathered beats): live entries (until their write goes out) from qw_f, the next to
  // write; qw_r the next whose fill read is due (qw_rn from qw_r on)
  logic [WW:0] qw_n [NCH], qw_rn [NCH];
  logic [WW-1:0] qw_f [NCH], qw_r [NCH];
  // the queue's data and fill data (u_qwd without WPOST, u_wrm with SW_FILL: CB bytes a slot),
  // block RAMs read ahead at the next write's next value (qw_fn); as LUT RAMs qw_f reached every
  // cell (the m8 card: 11,980 address pins each, over both channels)
  logic [WW-1:0] qw_fn [NCH];
  logic [WQD-1:0] wpart [NCH], wgot [NCH];  // per slot: bytes missing; its fill data is in
  // read-after-write hazards: whb counts, per bucket (a hash of the beat), the live entries past
  // the read point (from qw_f up to qw_r: fill read issued, or whole); wnz: whb != 0. A count is
  // read only while its bucket is live (wnz): the reset clears wnz alone (on core_rst's fan-out,
  // the counts' 2 x NH x (WW+1) flip-flops missed 100 MHz on 9.1 ns of route) and an increment
  // of a bucket that is not live starts it at 1
  logic [WW:0]  whb [NCH][NH];
  logic [NH-1:0] wnz [NCH];
  logic [NCH-1:0] h_inc, h_dec;             // slot qw_r passes the read point / qw_f's write goes
  // reads in flight per channel, in command order; the oldest one's tag registered (tgh)
  logic [TW:0]  tg_n [NCH];
  logic [TW-1:0] tg_h [NCH];
  tg_t          tgh [NCH], tg_in [NCH], tg_nx1 [NCH];
  // SW gather buffers (see the top): valid, beat and bytes written, cycles since the last merge.
  // Their data (gd) takes an SW write a cycle after it is taken (gw_*: taken last cycle, to the
  // buffer's beat), so the take (at the end of QUANT's TMEM grant) enables only gb's 89
  // flip-flops per channel, not the data's 512 (the 5e5a build's widest DMA -> memory group:
  // sleft -> gb data, 2048 endpoints); a gathered beat's data goes into the SW queue a cycle
  // after it is pushed (u_qwd)
  typedef struct packed {
    logic [CAW-1:0] m;
    logic [CB-1:0]  strb;
  } gs_t;
  // NG buffers (_q; SWG: one, its channel gc_q, the others unused); gv, gb, gd and gage are each
  // channel's view of its buffer (SWG: the one buffer, valid on its channel alone)
  localparam int NG = SWG ? 1 : NCH;
  logic [NCH-1:0] gv_q;
  gs_t          gb_q [NCH];
  logic [CB*8-1:0] gd_q [NCH];
  logic [2:0]   gage_q [NCH];
  logic [CL-1:0] gc_q;
  logic [NCH-1:0] gv;
  gs_t          gb [NCH];
  logic [CB*8-1:0] gd [NCH];
  logic [2:0]   gage [NCH];
  for (genvar c = 0; c < NCH; c++) begin : g_gv
    localparam int G = SWG ? 0 : c;
    assign gv[c] = gv_q[G] && (!SWG || gc_q == CL'(c));
    assign gb[c] = gb_q[G];
    assign gd[c] = gd_q[G];
    assign gage[c] = gage_q[G];
  end
  logic [NCH-1:0] gw_v;
  logic [WB-1:0] gw_word;
  logic [SB-1:0] gw_be;
  // the write's data, a copy per buffer's gd (each bit feeds 16 of a channel's flip-flops;
  // one copy for both, 32 loads over both channels' buffers, was the 812bb01 build's worst
  // native memory path: gw_wdata -> gd, 0 levels, 96% route, +0.175 at 133.33 MHz); keep: not
  // merged back into one
  (* keep = "true" *) logic [8*SB-1:0] gw_wdata [NG];

  // posted write data (WPOST: see the top). A take may be followed, before n_wfree shows it, by
  // the beat going in the cycle n_wfree was sampled, the B stage's and the SW beat's (they go in
  // this cycle or later) and this cycle's B take and SW push: PM beats a channel
  localparam int PM = 5;
  localparam int WOD = 32;                   // write-order entries a channel (posted beats whose
  localparam int WOW = $clog2(WOD);          //   writes have not gone out: in its FIFO, <= 32)
  logic [NCH-1:0] ok;                        // channel c's FIFO has PM beats free (registered)
  logic [NCH-1:0] sp;                        // the SW beat pushed last cycle goes in now (gd)
  logic [CB-1:0]  sps [NCH];                 //   with its bytes written (gb's last cycle; SWG:
                                             //   [0] alone)
  logic           bsv, bh;                   // a B write's beat staged; it waits this cycle (its
  logic           bh_q;                      //   channel's SW beat goes in; WHB: or the bus carries
                                             //   a high half): bh, at WHB 0 from a register (bh_q)
  logic [CL-1:0]  bsc;                       //   its channel, data
  logic [CB*8-1:0] bsd;                      //   and word enables
  logic [CB/4-1:0] bsm;
  logic [NCH-1:0] bst;                       // the staged B beat goes in on channel c now
  logic           gi;                        // a beat goes in now (an SW or the staged B beat)
  logic           whv;                       // WHB: the bus carries the last beat's high half
  logic [NCH-1:0] whc;                       //   on its channel, from a register (wh_d, wh_m)
  logic [CB*4-1:0] wh_d;
  logic [CB/2-1:0] wh_m;
  logic [CB*8-1:0] wpd;                      // SWG: the posted beat's data and byte mask, every
  logic [CB-1:0]  wpm;                       //   channel's (the SW beat's if one goes in, else B's)
  always_comb begin
    wpd = |sp ? gd_q[0] : bsd;
    for (int k = 0; k < CB / 4; k++) wpm[4 * k +: 4] = |sp ? sps[0][4 * k +: 4] : {4{bsm[k]}};
  end
  logic [WOD-1:0] wov [NCH];                 // per channel, in the FIFO's order, the posted beats
  logic [WOW:0]   won [NCH];                 //   whose writes have not gone out: B's (1) or SW's
  logic [WOW-1:0] woh [NCH];
  logic [NCH-1:0] wo1;                       // the head entry's, registered
  logic [NCH-1:0] wob, wos;                  // the next write to go out on c is B's / SW's

  // order of B reads (tags) and of A reads (channel, word, reuse)
  // B: the tag, the wide chunk's hash and the chunk's index in it (which channels hold which of
  // its beats); LUT RAM (a flat vector: see g_mem)
  typedef struct packed { logic [KB-1:0] k; logic [CL-1:0] h; logic tag; } bo_t;
  logic [$bits(bo_t)-1:0] bt_q [OD];
  logic [OW:0]  bt_n;
  logic         bt_nz, bt_n1, bt_n2, bt_g2;  // bt_n != 0, == 1, == 2, >= 2, registered with it
  logic [OW-1:0] bt_h;
  // A reads in order: channel, word, reuse of the last beat, run beats to drop before this one
  typedef struct packed { logic [CL-1:0] c; logic [WB-1:0] idx; logic reuse; logic [RNW-1:0] drop; } ao_t;
  ao_t          ao_q [OD];
  ao_t          aoh;                           // the oldest A read
  logic [OW:0]  ao_n;
  logic         ao_nz;                         // ao_n != 0, registered with it
  logic [OW-1:0] ao_h;
  // A beat reuse: the logical beat (bits 29:WB of the word address) and channel of the last A read
  logic           al_v;
  logic [CL-1:0]  al_c;
  logic [AW-3-WB:0] al_beat;

  always_comb begin
    b_rdy = bt_n < OD && (!WPOST || (&ok && !bh));
    for (int c = 0; c < NCH; c++) b_rdy &= qb_n[c] < QD;
  end
  // Port A's requests come from the slice's port A register (otpu_slice; this adapter's own, on
  // the way in, until it moved there): the slice's grant chain (DMA busy -> port arbitration ->
  // the MXU's issue -> the scale address) does not run on into the channel hash, the run compare
  // and the queue writes in one cycle.
  logic        a_req, a_we, a_rdy;
  logic [WA-1:0] a_addr;
  logic [31:0] a_wdata;
  logic [3:0]  a_be;
  assign a_req = a_req_x; assign a_we = a_we_x; assign a_addr = a_addr_x;
  assign a_wdata = a_wdata_x; assign a_be = a_be_x;
  always_comb begin
    a_rdy = ao_n < OD;
    sw_rdy = !WPOST || (&ok && !bh && !(WHB && gi));
`ifdef OTPU_MUT_WGI
    sw_rdy = !WPOST || (&ok && !bh);                              // (mutant: an SW take under a high half)
`endif
    w_rdy = HAS_W;
    for (int c = 0; c < NCH; c++) begin
      a_rdy &= qa_n[c] < QD;
      sw_rdy &= qw_n[c] < WQD;
      w_rdy &= qx_n[c] < QD;
    end
  end
  assign a_rdy_x = a_rdy;
  wire sw_take = sw_req && sw_rdy;
  wire [CL-1:0] sw_ch = chan_of(sw_addr);
  wire [CAW-1:0] sw_m = sw_addr[PWW +: CAW];
  wire b_take = b_req && b_rdy;
  wire a_take = a_req && a_rdy;
  wire w_take = w_req && w_rdy;
  // the W queues a W read goes into: every channel's, or (w_one) its chunk's channel's
  wire [CL-1:0] w_ch = chan_of(w_addr);
  logic [NCH-1:0] qx_push;
  always_comb
    for (int c = 0; c < NCH; c++) qx_push[c] = w_take && (!(XPOST && w_one) || w_ch == CL'(c));
  wire [CL-1:0] a_ch = chan_of(a_addr);
  // the channels of the chunk's beats: beat j (logical beat k x BPB + j of the wide chunk) on
  // channel (k x BPB + j) ^ h. b_addr comes at the end of the slice's grant chain (DMA request ->
  // the MXU's grant and request -> the port's address mux), and its 27-bit parity then ran into
  // every queue write (b_par: the slice muxes its sources' folds instead)
  wire [CL-1:0] b_h = CHASH ? b_par : CL'(0);
  wire [KB-1:0] b_k = (CL > BL) ? KB'(b_addr[PWW-1:0] >> (WB + BL)) : KB'(0);
  logic [NCH-1:0] b_in;                       // channel c holds one of the chunk's beats,
  logic [BL > 0 ? BL - 1 : 0:0] b_j [NCH];    //   its beat b_j
  logic [BPB-1:0] b_hnz;                      // the beats a B write writes
  always_comb
    for (int c = 0; c < NCH; c++) begin
      logic [CL-1:0] l;
      l = CL'(c) ^ b_h;
      b_in[c] = (CL > BL) ? (l >> BL) == CL'(b_k) : 1'b1;
      b_j[c] = (BL > 0) ? l : '0;
    end
  always_comb
    for (int j = 0; j < BPB; j++) b_hnz[j] = b_wmask[(CB / 4) * j +: CB / 4] != 0;
  wire b_wst = b_take && b_we && b_hnz[0];          // WPOST: a B write with a beat to write, and
  wire [CL-1:0] b_ch = CL'(b_k) ^ b_h;              //   its channel (BPB 1)
  wire [AW-3-WB:0] a_beat = a_addr[AW-3:WB];
  wire a_reuse = !a_we && al_v && al_beat == a_beat;
  // A runs per channel: the next channel beat of the run, valid (no write since), beats not
  // yet taken (in flight or stored)
  logic [CAW-1:0] pnx [NCH];
  logic [NCH-1:0] pv;
  logic [RNW-1:0] pfl [NCH];
  wire  [CAW-1:0] a_cb = a_addr[PWW +: CAW];      // the channel beat
  wire  a_hit = !a_we && !a_reuse && pv[a_ch] && a_cb == pnx[a_ch] && pfl[a_ch] != 0;

  // ------------------------------------------------------------------ response FIFOs
  // stored entries (RHB: half beats); beats stored + in flight (the room reserved)
  logic [REW:0] rb_n [NCH];
  logic [RW:0]  rb_res [NCH];
  logic [REW-1:0] rb_h [NCH], rb_t [NCH];
  // the B FIFO's memory in RBP pieces of HDW / RBP bits. Deep (RBB: RD >= 128, the cards' 128
  // beats of 512 bits), block RAMs read ahead (otpu_ram_1r1w_ra, u_rbm) at the head's next value
  // (rb_hn): as LUT RAMs (RAMD64Es) the head and the tail reached every cell (the card's m8:
  // 38,528 address pins), and the tail took a kept copy per piece; on block RAMs each reaches RBP
  // of them. Shallow (the WG targets' 32 beats of 1,024 bits: 16 RAMB36 a channel as block RAM, a
  // sixteenth of their depth used, against ~0.6K LUTs as LUT RAM), LUT RAM (rbm) read at the
  // head, each piece written at its own copy of the tail (rb_tq; keep: not merged back). One
  // tail drove the write addresses of all of a channel's RAMD64Es, spread over the columns its
  // read data crosses (110ec6d at 133.33 MHz: rb_t_rep -> rbm WADR, fanout 577, 0 levels, 6.9 ns
  // of route, +0.195 ns); rb_t stays as the copies' reference (the simulation check). RB_BLOCK:
  // block RAM at any depth (for adapters in a die short of LUT RAM sites: ~600 a channel in
  // the die's free RAMB36 instead)
  localparam int RBP = 8;
  localparam bit RBB = RD >= 128 || RB_BLOCK;
  logic [REW-1:0] rb_hn [NCH];
  (* keep = "true" *) logic [REW-1:0] rb_tq [NCH][RBP];
  logic [AEW:0] ra_n [NCH];                 // (as rb_n, rb_res)
  logic [AW_:0] ra_res [NCH];
  logic [AEW-1:0] ra_h [NCH], ra_t [NCH];
  logic [XW:0]  xi [NCH];                   // W reads in flight (command taken, data not back)
  // XPOST: the channels' posted-beat counts, two flops from their clock and decoded (p_cnt), and
  // the last count seen (pc_q): the beats posted since then return W reads in flight (xi)
  logic [PCW-1:0] pc_q [NCH];
  function automatic logic [PCW-1:0] g2b(input logic [PCW-1:0] g);
    logic [PCW-1:0] b;
    for (int i = 0; i < PCW; i++) b[i] = ^(g >> i);
    return b;
  endfunction
  if (XPOST) begin : g_pc
    (* ASYNC_REG = "TRUE" *) logic [NCH-1:0][PCW-1:0] pg1, pg2;
    always_ff @(posedge clk) begin
      pg1 <= p_cntg;
      pg2 <= pg1;
      for (int c = 0; c < NCH; c++) p_cnt[c] <= g2b(pg2[c]);
    end
  end else begin : g_npc
    assign p_cnt = '0;
  end

  // ------------------------------------------------------------------ queue and FIFO memories
  // (per channel; written in their own processes so they map to LUT RAM)
  logic [NCH-1:0] qb_push, qa_push, qw_push;
  qb_t          qb_e [NCH];
  qa_t          qa_e [NCH];
  qb_t          hb [NCH];
  qa_t          ha [NCH];
  qw_t          hw [NCH];
  logic [CAW-1:0] hx [NCH];
  logic [XSW-1:0] hxs [NCH];                       // XPOST: the W head's slot
  // slot qw_r (the read point): its beat, bucket and whether it is partial, registered (wadr_r,
  // wh_r, wp_r; see their process); the next slot's beat and bucket; the bucket of slot qw_f
  logic [CAW-1:0] wadr_r [NCH], wadr_n [NCH];
  logic [HW-1:0] wh_r [NCH], wh_n [NCH], wh_f [NCH];
  logic [NCH-1:0] wp_r;
`ifndef SYNTHESIS
  logic [CAW-1:0] wadr_o [NCH];
  logic [HW-1:0] wh_o [NCH];
`endif
  logic [HDW-1:0] rb_head [NCH], ra_head [NCH];
  logic [CB*8-1:0] wr_head [NCH];
  logic [NCH-1:0] a_hi;                            // RHB: channel c's head beat goes out (its read
                                                   //   port reads the high half while alv)
  logic [NCH-1:0] ra2;                             // ra_n >= 2, registered with it (RHB: a beat in)
  logic [NCH-1:0] rbz;                             // rb_n != 0, registered with it
  logic [NCH-1:0] rtk;                             // a read command taken (tag push)
  // RHB: the next half on n_rvalid is a beat's high one (rhh); rl: this one is a beat's last
  logic [NCH-1:0] rhh, rl;
  always_ff @(posedge clk) rhh <= (rst || !RHB) ? '0 : rhh ^ n_rvalid;
  assign rl = RHB ? rhh : '1;
  always_comb begin
    for (int c = 0; c < NCH; c++) begin
      // channel c holds the chunk's beat b_j[c], if any
      qb_push[c] = b_take && b_in[c] && (!b_we || b_hnz[b_j[c]]);
      qb_e[c].we = b_we;
      qb_e[c].m = b_addr[PWW +: CAW];
      qb_e[c].data = b_wdata[CB * 8 * b_j[c] +: CB * 8];
      qb_e[c].wmask = b_we ? b_wmask[(CB / 4) * b_j[c] +: CB / 4] : '0;
      qa_push[c] = a_take && a_ch == CL'(c) && !a_reuse && !a_hit;
      qa_e[c].we = a_we;
      qa_e[c].m = a_cb;
      qa_e[c].idx = a_addr[WB-1:0];
      qa_e[c].data = a_wdata;
      qa_e[c].be = a_be;
      // the gathered beat goes out: another beat's SW write (room is sure: sw_rdy), or, with
      // no SW write this cycle and room in the queue, full or idle
      // (WPOST: room in the channel's write-data FIFO, and not while a B beat waits)
      // (SWG: an SW write to another beat on any channel)
      qw_push[c] = gv[c] && (sw_take && (SWG || sw_ch == CL'(c)) ? (SWG && sw_ch != CL'(c)) || gb[c].m != sw_m
                   : qw_n[c] < WQD && (&gb[c].strb || gage[c] >= 3'(WGATHER)) &&
                     (!WPOST || (ok[c] && !bh && !(WHB && gi))));
    end
  end
  // the SW gather buffers' data: last cycle's write merged (see gd); last cycle's tail slot
  logic [WW-1:0] ql_t [NCH];
  logic [NCH-1:0] ql_v;
  always_ff @(posedge clk) begin
    for (int c = 0; c < NCH; c++) begin
      ql_t[c] <= WW'(qw_f[c] + qw_n[c]);
      ql_v[c] <= !qw_n[c][WW];
    end
    for (int g = 0; g < NG; g++) begin
      // a byte the beat's writes have not written (its strb bit clear) is never read: its write
      // goes out masked (SW_FILL 0) or merged with the fill read's (SW_FILL 1), so a byte
      // changes only when written (a flip-flop enable, no merge multiplexer; a new beat's
      // bytes keep the last beat's)
      if (gw_v[g])
        for (int k = 0; k < CB; k++)
          if ((gw_word >> SSL) == WB'(k / SB) && gw_be[k % SB])
            gd_q[g][8 * k +: 8] <= gw_wdata[g][8 * (k % SB) +: 8];
      gw_wdata[g] <= sw_wdata;
    end
    gw_word <= sw_addr[WB-1:0]; gw_be <= sw_be;
  end

  for (genvar c = 0; c < NCH; c++) begin : g_mem
    // flat vectors: Vivado builds a RAM of structs from registers
    // qbm: 2 QD slots for at most QD entries (b_rdy), so the slot at the tail is never live and
    // is written every cycle; a push only moves the tail. Its write enable was the push, which
    // comes at the end of the slice's grant chain and fanned out to all of the queue's LUT RAM
    // (qb_t's 554 bits: 745 cells per channel; the 959b425 build's worst DMA -> memory paths,
    // -0.390 ns at 133.33 MHz). At QD = 16 the RAM32Ms hold 32 slots already. qam the same: its
    // push (a_take at the end of the channel decode and the run compare, 12 levels from the
    // slice's port A register) ran into its write enables (the m8 card at a 150 MHz equivalent).
    // WPOST: the B and SW queues hold no write data (it went to the channel at the take): qbm the
    // command alone, qwm the beat alone, no u_qwd
    localparam int QBB = WPOST ? 1 + CAW : $bits(qb_t);
    localparam int QWB = WPOST ? CAW : $bits(gs_t);
    logic [QBB-1:0] qbm [2 * QD];
    logic [$bits(qa_t)-1:0] qam [2 * QD];
    logic [QWB-1:0] qwm [WQD];              // per slot: the beat and its bytes written
    logic [CB*8-1:0] hw_d;                  // and their data (u_qwd, see gd)
    logic [CAW-1:0] wam [WQD];              // per slot: the beat
    logic [HW-1:0] whm [WQD];               // per slot: its hazard bucket
    logic [$bits(tg_t)-1:0] tgm [TD];       // the reads in flight
    logic [HDW-1:0] ram [AE];
    always_ff @(posedge clk)
      qbm[(QW + 1)'(qb_h[c] + qb_n[c])] <= WPOST ? QBB'({qb_e[c].we, qb_e[c].m}) : QBB'(qb_e[c]);
    always_ff @(posedge clk) qam[(QW + 1)'(qa_h[c] + qa_n[c])] <= qa_e[c];
    // the SW queue's tail slot (not live unless the queue is full) takes the gathered beat every
    // cycle: its write enables are registers, not the push (an SW write's take comes at the end
    // of QUANT's TMEM grant; the 5e5a build's worst DMA -> memory paths ran into qwm's WE). The
    // data goes into last cycle's tail slot (ql_t, ql_v) a cycle later, as gd has it: a slot
    // pushed last cycle is at the read point or before it, and its data is read (hw) only past it
    always_ff @(posedge clk)
      if (!qw_n[c][WW]) qwm[WW'(qw_f[c] + qw_n[c])] <= WPOST ? QWB'(gb[c].m) : QWB'(gb[c]);
    if (!WPOST) begin : g_qwd
      otpu_ram_1r1w_ra #(.W(CB*8), .N(WQD), .STYLE("block")) u_qwd (
        .clk, .we(ql_v[c]), .wa(ql_t[c]), .wd(gd[c]), .ra_n(qw_fn[c]), .rd(hw_d));
    end else begin : g_nqwd
      assign hw_d = '0;
    end
    always_ff @(posedge clk) if (!qw_n[c][WW]) wam[WW'(qw_f[c] + qw_n[c])] <= gb[c].m;
    always_ff @(posedge clk) if (!qw_n[c][WW]) whm[WW'(qw_f[c] + qw_n[c])] <= whash(gb[c].m);
    always_ff @(posedge clk) if (rtk[c]) tgm[TW'(tg_h[c] + tg_n[c])] <= tg_in[c];
    assign tg_nx1[c] = tg_t'(tgm[TW'(tg_h[c] + 1'b1)]);
    assign wadr_n[c] = wam[WW'(qw_r[c] + 1'b1)];
    assign wh_n[c] = whm[WW'(qw_r[c] + 1'b1)];
`ifndef SYNTHESIS
    assign wadr_o[c] = wam[qw_r[c]];                // (the registers' reference)
    assign wh_o[c] = whm[qw_r[c]];
`endif
    assign wh_f[c] = whm[qw_f[c]];
    for (genvar k = 0; k < RBP; k++) begin : g_rb
      localparam int PB = HDW / RBP;
      if (RBB) begin : g_ra
        otpu_ram_1r1w_ra #(.W(PB), .N(RE), .STYLE("block")) u_rbm (
          .clk, .we(n_rvalid[c] && tgh[c].k == K_B), .wa(rb_t[c]), .wd(n_rdata[c][PB * k +: PB]),
          .ra_n(rb_hn[c]), .rd(rb_head[c][PB * k +: PB]));
      end else begin : g_lut
        logic [PB-1:0] rbm [RE];
        always_ff @(posedge clk)
          if (n_rvalid[c] && tgh[c].k == K_B) rbm[rb_tq[c][k]] <= n_rdata[c][PB * k +: PB];
        assign rb_head[c][PB * k +: PB] = rbm[rb_h[c]];
      end
    end
    always_ff @(posedge clk)
      if (n_rvalid[c] && tgh[c].k == K_A) ram[ra_t[c]] <= n_rdata[c][HDW-1:0];
    // the SW queue's fill data, per slot (SW_FILL: only a fill read returns K_W)
    if (SW_FILL) begin : g_wrm
      otpu_ram_1r1w_ra #(.W(CB*8), .N(WQD), .STYLE("block")) u_wrm (
        .clk, .we(n_rvalid[c] && tgh[c].k == K_W), .wa(tgh[c].slot), .wd(n_rdata[c]),
        .ra_n(qw_fn[c]), .rd(wr_head[c]));
    end else begin : g_nwrm
      assign wr_head[c] = '0;
    end
    assign hb[c] = WPOST ? qb_t'{we: qbm[qb_h[c]][QBB-1], m: CAW'(qbm[qb_h[c]]), data: '0, wmask: '0}
                         : qb_t'($bits(qb_t)'(qbm[qb_h[c]]));
    assign ha[c] = qa_t'(qam[qa_h[c]]);
    gs_t hs;
    assign hs = WPOST ? gs_t'{m: CAW'(qwm[qw_f[c]]), strb: '0} : gs_t'($bits(gs_t)'(qwm[qw_f[c]]));
    assign hw[c] = qw_t'{m: hs.m, data: hw_d, strb: hs.strb};
    if (RHB) begin : g_rah                 // the head beat's low half, or (alv) its high half
      // (a_hi needs alv, and only alo, while empty, takes the low half: the address comes from
      // channel c's own registers, not from the oldest read's channel through a_out)
      assign ra_head[c] = ram[AEW'(ra_h[c] + AEW'(alv[c]))];
    end else begin : g_raw                 // the beat past the dropped ones
      assign ra_head[c] = ram[AW_'(ra_h[c] + AW_'(aoh.drop))];
    end
    if (HAS_W) begin : g_x                  // port W: its queue (the channel beat) and its beats
      localparam int XB = XPOST ? CAW + XSW : CAW;   // (XPOST: the slot too)
      logic [XB-1:0] qxm [QD];
      always_ff @(posedge clk)
        if (qx_push[c]) qxm[QW'(qx_h[c] + qx_n[c])] <= XPOST ? XB'({w_slot, w_addr[PWW +: CAW]})
                                                         : XB'(w_addr[PWW +: CAW]);
      assign hx[c] = CAW'(qxm[qx_h[c]]);
      assign hxs[c] = XPOST ? XSW'(qxm[qx_h[c]] >> CAW) : '0;
      // a W read's beat goes out through a register (w_bv, w_bd: the MXU's chunk FIFO has its
      // room), as the B and A data leave theirs: from flip-flops (XPOST: the channel posts it)
      always_ff @(posedge clk) begin
        w_bv[c] <= !rst && !XPOST && n_rvalid[c] && tgh[c].k == K_X;
        if (!XPOST && n_rvalid[c] && tgh[c].k == K_X) w_bd[CB * 8 * c +: CB * 8] <= n_rdata[c];
      end
    end else begin : g_nx
      assign hx[c] = '0;
      assign hxs[c] = '0;
      assign w_bv[c] = 1'b0;
      assign w_bd[CB * 8 * c +: CB * 8] = '0;
    end
  end

  // ------------------------------------------------------------------ per-channel issue
  logic [NCH-1:0] hv;                            // a command shown and not taken (or a write
  logic [2:0]   hsrc [NCH];                      //   not yet out): shown again; its source
  logic [NCH-1:0] c_done, d_done;                // the shown write's command / data taken
  logic [NCH-1:0] sv;                            // a command is shown
  logic [2:0]   src [NCH];                       // its source (S_*)
  logic [NCH-1:0] swr;                           // it is a write
  logic [NCH-1:0] ctk, dtk;                      // command / write data taken this cycle
  logic [NCH-1:0] wiss;                          // a write goes out (both taken)
  logic [NCH-1:0] w_blk, w_hold;                 // the fill read waits (an older write of its
                                                 //   beat); the SW writes wait
  always_comb begin
    for (int c = 0; c < NCH; c++) begin
      logic e_a, e_wr, e_ww, e_x, e_b;
      // the SW queue's next partial beat: blocked while an older live entry may have its
      // address (one in its bucket; a hash collision only delays the read)
      w_blk[c] = wnz[c][wh_r[c]];
      // the queue's writes wait while its fill reads still go out (not blocked), unless half full
      w_hold[c] = (qw_rn[c] != 0) && wp_r[c] && !w_blk[c] && qw_n[c] < (WW + 1)'(WQD / 2);
      // A: a read's run (response room for the whole run as it starts), or a write
      e_a = (qa_n[c] != 0) && (ha[c].we || a_iss[c] != 0 || ra_res[c] <= (AW_ + 1)'(AD - APF));
      e_wr = (qw_rn[c] != 0) && wp_r[c] && !w_blk[c];
      // an SW write once its slot is past the read point and, if partial, has its fill data
      // (WPOST: a write goes out in its data's order: wos, wob)
      e_ww = (qw_n[c] != qw_rn[c]) && (!wpart[c][qw_f[c]] || wgot[c][qw_f[c]]) && !w_hold[c] &&
             (!WPOST || wos[c]);
      e_x = HAS_W && (qx_n[c] != 0) && xi[c] < (XW + 1)'(XD);
      e_b = (qb_n[c] != 0) && (hb[c].we ? !WPOST || wob[c] : rb_res[c] < (RW + 1)'(RD));
`ifdef OTPU_MUT_WORDER
      e_b = (qb_n[c] != 0) && (hb[c].we || rb_res[c] < (RW + 1)'(RD));   // (mutant: B writes out
`endif                                                                  //   of their data's order)
      sv[c] = hv[c] || e_a || e_wr || e_ww || e_x || e_b;
      src[c] = hv[c] ? hsrc[c] : e_a ? S_A : e_wr ? S_WR : e_ww ? S_WW
             : (e_x && !e_b) ? S_X : S_B;
      swr[c] = (src[c] == S_A && ha[c].we) || src[c] == S_WW || (src[c] == S_B && hb[c].we);
      n_cvalid[c] = sv[c] && !(swr[c] && c_done[c]);
      n_wvalid[c] = WPOST ? sp[c] || bst[c] || (WHB && whv && whc[c]) : sv[c] && swr[c] && !d_done[c];
      n_cwe[c] = swr[c];
      case (src[c])
        S_A:  n_caddr[c] = ha[c].m + CAW'(a_iss[c]);
        S_WR: n_caddr[c] = wadr_r[c];
        S_WW: n_caddr[c] = hw[c].m;
        S_X:  n_caddr[c] = hx[c];
        default: n_caddr[c] = hb[c].m;
      endcase
      n_wdata[c] = '0;
      n_wmask[c] = '0;
      if (WPOST && SWG && WHB) begin           // (WHB: the low half, or last beat's high half)
        n_wdata[c] = (CB * 8)'(whv ? wh_d : wpd[CB*4-1:0]);
        n_wmask[c] = CB'(whv ? wh_m : wpm[CB/2-1:0]);
      end else if (WPOST && SWG) begin         // (SWG: one bus, the same on every channel)
        n_wdata[c] = wpd;
        n_wmask[c] = wpm;
      end else if (WPOST) begin                // the posted beat: the SW beat's, else B's
        n_wdata[c] = sp[c] ? gd[c] : bsd;
        for (int k = 0; k < CB / 4; k++)
          n_wmask[c][4 * k +: 4] = sp[c] ? sps[c][4 * k +: 4] : {4{bsm[k]}};
      end else if (src[c] == S_WW && SW_FILL) begin
        for (int k = 0; k < CB; k++)
          n_wdata[c][8 * k +: 8] = hw[c].strb[k] ? hw[c].data[8 * k +: 8] : wr_head[c][8 * k +: 8];
        n_wmask[c] = '1;
      end else if (src[c] == S_WW) begin
        n_wdata[c] = hw[c].data;
        n_wmask[c] = hw[c].strb;
      end else if (src[c] == S_A) begin
        n_wdata[c][32 * ha[c].idx +: 32] = ha[c].data;
        n_wmask[c][4 * ha[c].idx +: 4] = ha[c].be;
      end else begin
        n_wdata[c] = hb[c].data;
        for (int k = 0; k < CB / 4; k++) n_wmask[c][4 * k +: 4] = {4{hb[c].wmask[k]}};
      end
      ctk[c] = n_cvalid[c] && n_cready[c];
      dtk[c] = !WPOST && n_wvalid[c] && n_wready[c];
      wiss[c] = sv[c] && swr[c] && (c_done[c] || ctk[c]) && (WPOST || d_done[c] || dtk[c]);
      // (XPOST: a W read is not tagged, its beats do not come back here)
      rtk[c] = ctk[c] && !swr[c] && !(XPOST && src[c] == S_X);
      tg_in[c].k = src[c] == S_A ? K_A : src[c] == S_WR ? K_W : src[c] == S_X ? K_X : K_B;
      tg_in[c].slot = qw_r[c];
      n_csrc[c] = tg_in[c].k;
      n_cslot[c] = (XPOST && src[c] == S_X) ? hxs[c] : '0;
`ifdef OTPU_MUT_XSLOT
      if (src[c] == S_X) n_cslot[c] = hxs[c] + 1'b1;   // (mutant: the beat into the next entry)
`endif
    end
  end

  // ------------------------------------------------------------------ posted write data (WPOST)
  // The B stage takes a B write's beat (bsv) unless its beat waits (bh, its enable: a register);
  // the beat goes in the next cycle, a cycle later where an SW beat goes in on its channel then
  // (bh; nothing is taken while it waits, so it goes in after it). The SW beat pushed goes in the
  // cycle after its push, when gd holds it whole (see gd) with gb's bytes of the push (sps). Each
  // beat that goes in enters its channel's write order unless its write goes out in that cycle
  // (with nothing ahead of it); a write that goes out from the order leaves it.
  // WHB: a beat that goes in takes the bus for two cycles, its high half the second (whv, from
  // wh_d / wh_m); nothing goes in then: an SW push or take waits for a cycle without a beat going
  // in (sw_rdy, qw_push: the SW beat pushed goes in the next cycle), the B stage for one without
  // the SW beat or a high half (bh, from registers)
  assign bh = WHB ? bsv && (|sp || whv) : bh_q;
  always_comb begin
    for (int c = 0; c < NCH; c++) begin
      bst[c] = bsv && !bh && bsc == CL'(c);
      wob[c] = won[c] != 0 ? wo1[c] : bst[c];
      wos[c] = won[c] != 0 ? !wo1[c] : sp[c];
    end
    gi = |sp || |bst;
  end
  always_ff @(posedge clk) begin
    bsv <= !rst && WPOST && (bh || b_wst);
    bh_q <= !rst && WPOST && !bh_q && b_wst && (SWG ? |qw_push : qw_push[b_ch]);
    sp <= rst || !WPOST ? '0 : qw_push;
    whv <= !rst && WHB && gi;
    whc <= sp | bst;
    wh_d <= wpd[CB*8-1:CB*4];
    wh_m <= wpm[CB-1:CB/2];
    if (!bh) begin
      bsd <= b_wdata[CB * 8 - 1:0];
      bsm <= b_wmask[CB / 4 - 1:0];
      bsc <= b_ch;
    end
    for (int c = 0; c < NCH; c++) begin
      logic push, pop;
      if (!SWG || c == 0) sps[c] <= gb[c].strb;
      ok[c] <= !rst && n_wfree[c] >= WFW'(PM);
`ifdef OTPU_MUT_CREDIT
      ok[c] <= !rst && n_wfree[c] != 0;                                 // (mutant: a beat's room only)
`endif
      push = (sp[c] || bst[c] || (!WPOST && n_wvalid[c])) && !(wiss[c] && won[c] == 0);
      pop = wiss[c] && won[c] != 0;
      if (push) wov[c][WOW'(woh[c] + won[c])] <= !sp[c];
      if (won[c] == 0 || (pop && won[c] == (WOW + 1)'(1))) wo1[c] <= !sp[c];
      else if (pop) wo1[c] <= wov[c][WOW'(woh[c] + 1'b1)];
      if (rst || !WPOST) begin
        won[c] <= '0; woh[c] <= '0;
      end else begin
        won[c] <= (WOW + 1)'(updn(16'(won[c]), push, pop));
        if (pop) woh[c] <= woh[c] + 1'b1;
      end
    end
  end

  // a write that went out on channel c (last cycle: wk_v, its beat wk_m) drops c's A run, or the
  // reused beat, when its beat is in them (see the top); a cycle late, off the path from the
  // command select and the channel's readies, and still before any A read that depends on the
  // write is taken (that comes after wr_idle, and waits a cycle in the A register)
  logic [NCH-1:0] wk_v, wk_r, wk_a;
  logic [CAW-1:0] wk_m [NCH];
  always_ff @(posedge clk) begin
    wk_v <= rst ? '0 : wiss;
    for (int c = 0; c < NCH; c++) wk_m[c] <= n_caddr[c];
  end
  always_comb
    for (int c = 0; c < NCH; c++) begin
      wk_r[c] = wk_v[c] && pv[c] && CAW'(wk_m[c] - pnx[c]) < CAW'(pfl[c]);
      wk_a[c] = wk_v[c] && al_v && al_c == CL'(c) && al_beat[AW-3-WB:CL] == wk_m[c];
    end

  // ------------------------------------------------------------------ merge
  // B: the oldest read's channels (bth: which hold its beats) each have its beat (RHB: its next
  // half; b_pl takes the low half into b_lo, b_out the chunk, a cycle apart at the least). RHB: in
  // the cycle a chunk goes out the next read (bt2), on another channel with its low half in, takes
  // that half into b_lo (b_ov), so reads on different channels go out a chunk a cycle; on the same
  // channel the FIFO's read port has the chunk's high half, and the next read loads after it
  // bth and bt2: the order FIFO's head and next entries in registers (as aoh), loaded with the
  // entries they hold next (bt_q[bt_h] and bt_q[bt_h + 1] after this cycle's pop and push: the
  // next entry or the one after it, bt3, or the entry pushed into the slot they move to), so the
  // LUT RAM's read stays out of b_out and b_lo's enable (B1: bt_h -> bt_q -> b_lo CE, 8 levels)
  bo_t bth, bt2, bt3, b_new;
  assign bt3 = bo_t'(bt_q[OW'(bt_h + 2'd2)]);
  assign b_new = '{k: b_k, h: b_h, tag: b_tag};     // (the tail slot's write: live if pushed)
  always_ff @(posedge clk) begin
    bth <= b_out ? (bt_n1 ? b_new : bt2) : (!bt_nz ? b_new : bth);
    bt2 <= b_out ? (bt_n2 ? b_new : bt3) : (bt_n1 ? b_new : bt2);
  end
  logic [NCH-1:0] bt_in;
  logic b_out, b_any, b_pl, b_ov;
  logic [CL-1:0] b_c, b_c2, b_lc;                // RHB: the oldest read's channel, the next one's,
                                                 //   b_lo's (b_pl: the oldest's; b_ov: the next's)
  logic bph;                                     // RHB: the oldest read's low half is in b_lo
  logic [HDW-1:0] b_lo;
  always_comb begin
    b_any = bt_nz;
    for (int c = 0; c < NCH; c++) begin
      bt_in[c] = (CL > BL) ? ((CL'(c) ^ bth.h) >> BL) == CL'(bth.k) : 1'b1;
      if (bt_in[c] && !rbz[c]) b_any = 1'b0;
    end
    b_out = b_any && (!RHB || bph);
    b_pl = RHB && b_any && !bph;
    b_c = CL'(bth.k) ^ bth.h;
    b_c2 = CL'(bt2.k) ^ bt2.h;
    b_ov = RHB && b_out && bt_g2 && b_c2 != b_c && rbz[b_c2];
    b_lc = bph ? b_c2 : b_c;
`ifdef OTPU_MUT_BOV
    b_ov = RHB && b_out && bt_g2 && rbz[b_c2];    // (mutant: on the same channel)
`endif
  end
  logic [NCH-1:0] b_pop;                         // channel c's B FIFO pops an entry
  always_comb
    for (int c = 0; c < NCH; c++) b_pop[c] = (b_out || b_pl) && bt_in[c] || b_ov && b_c2 == CL'(c);
  always_ff @(posedge clk) begin
    if (rst || !RHB) bph <= 1'b0;
    else if (b_pl) bph <= 1'b1;
    else if (b_out) bph <= b_ov;
    if (b_pl || b_ov) b_lo <= rb_head[b_lc];
  end
  // the B FIFO's head's next value, as the state block below moves it (a pop of the channels
  // that hold the oldest read's beats)
  for (genvar c = 0; c < NCH; c++) begin : g_rbh
    assign rb_hn[c] = rst ? '0 : b_pop[c] ? REW'(rb_h[c] + 1'b1) : rb_h[c];
  end
  // the SW queue's next write's next value (its pop: popw in the state block)
  for (genvar c = 0; c < NCH; c++) begin : g_qwf
    assign qw_fn[c] = rst ? '0 : (src[c] == S_WW && wiss[c]) ? WW'(qw_f[c] + 1'b1) : qw_f[c];
  end
  // the oldest A read, registered (the order FIFO's LUT RAM read stays out of the run-beat and
  // word select into a_rdata): after a pop the next entry (written at least a cycle before), or
  // the entry pushed into an empty queue
  ao_t          ao_nx1, ao_in;
  assign ao_nx1 = ao_q[OW'(ao_h + 1'b1)];
  assign ao_in = '{c: a_ch, idx: a_addr[WB-1:0], reuse: a_reuse,
                   drop: (a_reuse || a_hit) ? RNW'(0) : pfl[a_ch]};
  logic [CB*8-1:0] a_last;                       // the beat of the last fetched A read
  // RHB: the oldest read goes out (a_out) with its channel's head beat whole (alo has its low
  // half, the high half is in), or first skips the run beats it drops (a_skip, once they are in);
  // each channel's read port reads its head's high half as the beat goes out (a_hi), else its
  // low half, into alo while that is empty (a_ld)
  logic [NCH-1:0] alv, a_ld;
  logic [HDW-1:0] alo [NCH];
  // (ao_nz and ra2: the registered compares, so a_out's enable of a_rdata is a select of
  // registers, not of counters through compares)
  wire a_skip = RHB && ao_nz && !aoh.reuse && aoh.drop != 0 &&
                ra_n[aoh.c] >= (AEW + 1)'(2 * aoh.drop);
  wire a_out = ao_nz && (aoh.reuse || (RHB ? aoh.drop == 0 && alv[aoh.c] && ra2[aoh.c]
                                           : ra_n[aoh.c] > (AW_ + 1)'(aoh.drop)));
  always_comb
    for (int c = 0; c < NCH; c++) begin
      a_hi[c] = RHB && a_out && !aoh.reuse && aoh.c == CL'(c);
      a_ld[c] = RHB && !alv[c] && ra_n[c] != 0 && !a_hi[c] && !(a_skip && aoh.c == CL'(c));
    end
  always_ff @(posedge clk)
    for (int c = 0; c < NCH; c++) begin
      if (rst || !RHB) alv[c] <= 1'b0;
      else if (a_hi[c] || (a_skip && aoh.c == CL'(c))) alv[c] <= 1'b0;
      else if (a_ld[c]) alv[c] <= 1'b1;
      // alo takes the read port every cycle it is empty (alv 0: its value unused until a_ld
      // sets alv, in the cycle it takes the head's low half), so its enable is alv alone
      if (!alv[c]) alo[c] <= ra_head[c];
    end
  wire [CB*8-1:0] a_src = aoh.reuse ? a_last : (CB * 8)'(ra_head[aoh.c]);
  always_ff @(posedge clk)
    if (a_out) aoh <= (ao_n == (OW + 1)'(1)) ? ao_in : ao_nx1;
    else if (!ao_nz) aoh <= ao_in;
    else if (a_skip) aoh.drop <= '0;
  // the oldest read in flight's tag, registered likewise (it steers the response FIFO writes)
  always_ff @(posedge clk)
    for (int c = 0; c < NCH; c++)
      if (n_rvalid[c] && rl[c]) tgh[c] <= (tg_n[c] == (TW + 1)'(1)) ? tg_in[c] : tg_nx1[c];
      else if (tg_n[c] == 0) tgh[c] <= tg_in[c];
  assign b_rvalid = b_out;
  assign b_rtag = bth.tag;
  // beat j of the chunk: logical beat k x BPB + j of the wide chunk, on that ^ h
  always_comb
    if (RHB) b_rdata = (D * 8)'({rb_head[b_c], b_lo});   // (BPB 1)
    else
      for (int j = 0; j < BPB; j++)
        b_rdata[CB * 8 * j +: CB * 8] = (CB * 8)'(rb_head[(CL'(bth.k) << BL | CL'(j)) ^ bth.h]);
  // the A read data registered on the way out (the head's run-beat and word select ran into the
  // MXU's scale FIFO block RAM in one cycle). AB = CB: the register holds the beat it last
  // fetched, so a read that reuses it presents it again by holding (no a_last, no reuse
  // multiplexer; the data counts only with a_rvalid)
  always_ff @(posedge clk) begin
    a_rvalid <= !rst && a_out;
    if (AB == CB) begin
      if (a_out && !aoh.reuse) a_rdata <= RHB ? (AB * 8)'({ra_head[aoh.c], alo[aoh.c]})
                                              : (AB * 8)'(ra_head[aoh.c]);
    end else a_rdata <= (AB * 8)'(a_src[256 * aoh.idx[WB-1:3] +: 256]);
  end

  // the tag FIFO writes its tail every cycle too (the slot is not live unless the FIFO is full),
  // so its write enable is a register, not b_take
  always_ff @(posedge clk) begin
    if (!bt_n[OW]) bt_q[OW'(bt_h + bt_n)] <= {b_k, b_h, b_tag};
    if (a_take && !a_we)
      ao_q[OW'(ao_h + ao_n)] <= ao_in;
  end

  // ------------------------------------------------------------------ writes outstanding
  // wq_n counts the writes (beats) accepted before the last cycle that have not gone out; wacc_q
  // holds the last cycle's accepts (folded in a cycle late, off the rdy -> take path). A write
  // goes out at least a cycle after its accept, so wq_n never goes negative. iss_w[c] counts
  // channel c's writes gone out (mod 2^16), which n_wdone reaches once the controller has them;
  // it starts at n_wdone's value in reset (the channel module's count need not reset with rst,
  // as long as no write of ours is still on its way to the controller then).
  logic [PW-1:0] wq_n;
  logic [NCH+BPB:0] wacc_q;
  logic [15:0]   iss_w [NCH];
  always_comb begin
    wr_idle = (wq_n == 0) && (wacc_q == '0) && (gv == '0);
    for (int c = 0; c < NCH; c++) wr_idle &= iss_w[c] == n_wdone[c];
  end
  // (W: a read is in its queue, in flight, or its beat in the output register)
  always_comb begin
    rd_idle = (bt_n == 0) && (ao_n == 0);
    for (int c = 0; c < NCH; c++) rd_idle &= qx_n[c] == 0 && xi[c] == 0 && !w_bv[c];
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      for (int c = 0; c < NCH; c++) begin
        qb_n[c] <= '0; qb_h[c] <= '0; qa_n[c] <= '0; qa_h[c] <= '0; a_iss[c] <= '0;
        qx_n[c] <= '0; qx_h[c] <= '0;
        qw_n[c] <= '0; qw_f[c] <= '0; qw_r[c] <= '0; qw_rn[c] <= '0; wnz[c] <= '0;
        tg_n[c] <= '0; tg_h[c] <= '0;
        rb_n[c] <= '0; rb_res[c] <= '0; rb_h[c] <= '0; rb_t[c] <= '0;
        if (!RBB) for (int k = 0; k < RBP; k++) rb_tq[c][k] <= '0;
        ra_n[c] <= '0; ra_res[c] <= '0; ra_h[c] <= '0; ra_t[c] <= '0; ra2[c] <= 1'b0;
        rbz[c] <= 1'b0;
        xi[c] <= '0;
        pc_q[c] <= p_cnt[c];
        iss_w[c] <= n_wdone[c];
        pfl[c] <= '0; gage_q[c] <= '0;
      end
      bt_n <= '0; bt_h <= '0; ao_n <= '0; ao_nz <= 1'b0; ao_h <= '0;
      bt_nz <= 1'b0; bt_n1 <= 1'b0; bt_n2 <= 1'b0; bt_g2 <= 1'b0;
      al_v <= 1'b0;
      pv <= '0;
      wq_n <= '0; wacc_q <= '0;
      gv_q <= '0; gw_v <= '0;
      hv <= '0; c_done <= '0; d_done <= '0;
    end else begin
      // writes accepted (the take strobes: an A or an SW write goes to exactly one channel, so
      // a_reuse and the channel decode stay off this path; a B write to each channel whose beat
      // it writes) and gone out
      begin
        logic [PW-1:0] n;
        n = wq_n;
        for (int k = 0; k <= NCH + BPB; k++) n = n + PW'(wacc_q[k]);
        for (int c = 0; c < NCH; c++) n = n - PW'(wiss[c]);
        wq_n <= n;
      end
      wacc_q <= {qw_push, a_take && a_we, {BPB{b_take && b_we}} & b_hnz};
      for (int c = 0; c < NCH; c++) begin
        logic [QW:0] na;
        logic [REW:0] rbn;
        logic [RW:0] rbr;
        logic [AEW:0] ran;
        logic [AW_:0] rar;
        logic [XW:0] nxi;
        logic [TW:0] ntg;
        logic popb, popa, popw, popx;
        na = qa_n[c]; ntg = tg_n[c];
        rbn = rb_n[c]; rbr = rb_res[c]; ran = ra_n[c]; rar = ra_res[c];
        nxi = xi[c];
        // ---- accept
        if (qa_push[c]) na = na + 1;
        if (!qw_n[c][WW]) begin                // the tail slot, as qwm (see g_mem)
          logic [WW-1:0] t;
          t = WW'(qw_f[c] + qw_n[c]);
          wpart[c][t] <= SW_FILL && !(&gb[c].strb);
          wgot[c][t] <= 1'b0;
        end
        // ---- the command stream: a shown command is shown again until taken, a write until
        // its command and its data are taken
        hv[c] <= sv[c] && (swr[c] ? !wiss[c] : !ctk[c]);
        hsrc[c] <= src[c];
        c_done[c] <= swr[c] && !wiss[c] && (c_done[c] || ctk[c]);
        d_done[c] <= swr[c] && !wiss[c] && (d_done[c] || dtk[c]);
        popa = src[c] == S_A && (swr[c] ? wiss[c] : ctk[c] && a_iss[c] == RNW'(APF - 1));
        popb = src[c] == S_B && (swr[c] ? wiss[c] : ctk[c]);
        popw = src[c] == S_WW && wiss[c];
        popx = src[c] == S_X && ctk[c];
        if (rtk[c]) ntg = ntg + 1;
        if (src[c] == S_A && rtk[c]) begin      // an A run's beat; the run reserves its room
          a_iss[c] <= (a_iss[c] == RNW'(APF - 1)) ? RNW'(0) : a_iss[c] + 1'b1;
          if (a_iss[c] == 0) rar = rar + (AW_ + 1)'(APF);
        end
        if (src[c] == S_B && rtk[c]) rbr = rbr + 1;
`ifndef OTPU_MUT_XIDLE
        if (popx) nxi = nxi + 1;
        if (XPOST) nxi = nxi - (XW + 1)'(PCW'(p_cnt[c] - pc_q[c]));   // (beats posted since)
`endif                                    // (mutant: rd_idle not waiting for the posted beats)
        pc_q[c] <= p_cnt[c];
        // a fill read passes the read point; a whole beat needs no fill read (e_wr is never set
        // for one): it passes the read point at once, so before its write can go (h_inc: either)
        if (h_inc[c]) qw_r[c] <= qw_r[c] + 1;
        if (popb) qb_h[c] <= qb_h[c] + 1;
        if (popa) begin qa_h[c] <= qa_h[c] + 1; na = na - 1; end
        if (popw) qw_f[c] <= qw_f[c] + 1;
        if (popx) qx_h[c] <= qx_h[c] + 1;
        if (wiss[c]) iss_w[c] <= iss_w[c] + 1'b1;
        // ---- read data, routed by the oldest read's tag
        // (RHB: each half an entry of the B or A FIFO; the tag moves on the beat's last half)
        if (n_rvalid[c]) begin
          case (tgh[c].k)
            K_B: begin
              rb_t[c] <= rb_t[c] + 1; rbn = rbn + 1;
              if (!RBB) for (int k = 0; k < RBP; k++) rb_tq[c][k] <= rb_tq[c][k] + 1;
            end
            K_A: begin ra_t[c] <= ra_t[c] + 1; ran = ran + 1; end
            K_X: if (rl[c]) nxi = nxi - 1;
            default: if (rl[c]) wgot[c][tgh[c].slot] <= 1'b1;   // the SW fill read: its slot has the data
          endcase
          if (rl[c]) begin
            tg_h[c] <= tg_h[c] + 1;
            ntg = ntg - 1;
          end
        end
        // ---- merge pops
        if (b_pop[c]) begin                        // (RHB: a half; the room at the chunk's)
          rb_h[c] <= rb_h[c] + 1;
          rbn = rbn - 1;
          if (b_out && bt_in[c]) rbr = rbr - 1;
        end
        if (RHB) begin                             // a beat out, or the run beats it drops
          if (a_hi[c]) begin
            ra_h[c] <= ra_h[c] + AEW'(2);
            ran = ran - (AEW + 1)'(2); rar = rar - 1'b1;
          end else if (a_skip && aoh.c == CL'(c)) begin
            ra_h[c] <= ra_h[c] + AEW'(2 * aoh.drop);
`ifdef OTPU_MUT_ASKIP
            ra_h[c] <= ra_h[c] + AEW'(2 * aoh.drop - 1);              // (mutant: a skip a half short)
`endif
            ran = ran - (AEW + 1)'(2 * aoh.drop); rar = rar - (AW_ + 1)'(aoh.drop);
          end
        end else if (a_out && !aoh.reuse && aoh.c == CL'(c)) begin    // (and the dropped run beats)
          ra_h[c] <= ra_h[c] + AW_'(aoh.drop) + 1'b1;
          ran = ran - (AW_ + 1)'(aoh.drop) - 1'b1;
          rar = rar - (AW_ + 1)'(aoh.drop) - 1'b1;
        end
        // ---- hazard buckets: slot qw_r passes the read point, slot qw_f's write goes out
        for (int k = 0; k < NH; k++)
          if (h_ik[c][k] || h_dk[c][k]) wnz[c][k] <= h_iv[c][k] || whb[c][wh_f[c]] != (WW + 1)'(1);
        qb_n[c] <= (QW + 1)'(updn(16'(qb_n[c]), qb_push[c], popb));
        qx_n[c] <= (QW + 1)'(updn(16'(qx_n[c]), qx_push[c], popx));
        qw_n[c] <= (WW + 1)'(updn(16'(qw_n[c]), qw_push[c], popw));
        qw_rn[c] <= (WW + 1)'(updn(16'(qw_rn[c]), qw_push[c], h_inc[c]));
        qa_n[c] <= na; tg_n[c] <= ntg;
        rb_n[c] <= rbn; rb_res[c] <= rbr; ra_n[c] <= ran; ra_res[c] <= rar;
        ra2[c] <= ran >= (AEW + 1)'(2);
        rbz[c] <= rbn != 0;
        xi[c] <= nxi;
      end
      // ---- order FIFOs
      begin
        logic [OW:0] aon, btn;
        aon = ao_n;
        if (b_out) bt_h <= bt_h + 1;
        if (a_take && !a_we) aon = aon + 1;
        if (a_out) begin
          ao_h <= ao_h + 1; aon = aon - 1;
          if (!RHB && !aoh.reuse) a_last <= (CB * 8)'(ra_head[aoh.c]);
        end
        btn = (OW + 1)'(updn(16'(bt_n), b_take && !b_we, b_out));
        bt_n <= btn;
        bt_nz <= btn != 0; bt_n1 <= btn == 1; bt_n2 <= btn == 2; bt_g2 <= btn >= 2;
        ao_n <= aon;
        ao_nz <= aon != 0;
      end
      // ---- SW gather: merge into the beat, or start a new one (the old one was pushed)
      // (SWG: the one buffer takes every channel's writes and holds its channel)
      for (int g = 0; g < NG; g++) begin
        logic take;
        take = sw_take && (SWG || sw_ch == CL'(g));
        gw_v[g] <= take;
        if (take) begin
          logic same;
          same = gv[SWG ? sw_ch : CL'(g)] && gb_q[g].m == sw_m;
          gv_q[g] <= 1'b1;
          gage_q[g] <= '0;
          gb_q[g].m <= sw_m;
          if (SWG) gc_q <= sw_ch;
          for (int k = 0; k < CB; k++)
            gb_q[g].strb[k] <= ((sw_addr[WB-1:0] >> SSL) == WB'(k / SB) && sw_be[k % SB]) ||
                               (same && gb_q[g].strb[k]);
        end else if (SWG ? |qw_push : qw_push[g]) begin
          gv_q[g] <= 1'b0;
        end else if (gv_q[g] && gage_q[g] != '1) begin
          gage_q[g] <= gage_q[g] + 1;
        end
      end
      // ---- A runs: a hit takes the next beat, a miss starts a run (dropping the rest)
      if (a_take && !a_we && !a_reuse) begin
        if (a_hit) begin
          pnx[a_ch] <= pnx[a_ch] + 1;
          pfl[a_ch] <= pfl[a_ch] - 1;
        end else begin
          pnx[a_ch] <= a_cb + 1;
          pfl[a_ch] <= RNW'(APF - 1);
          pv[a_ch] <= 1'b1;
        end
      end
      for (int c = 0; c < NCH; c++) if (wk_r[c]) pv[c] <= 1'b0;
      if (a_take && a_we) pv <= '0;
      // ---- A beat reuse: the beat of the last A read, forgotten on an A write or a write to it
      if ((a_take && a_we) || (|wk_a)) al_v <= 1'b0;
      else if (a_take && !a_we) begin
        al_v <= 1'b1;
        al_beat <= a_beat;
        al_c <= a_ch;
      end
      // ---- the host may have written since (last: an A read taken now decided with the old
      // state, and its run is dropped too)
      if (a_flush) begin
        pv <= '0;
        al_v <= 1'b0;
      end
    end
  end

  // the hazard buckets' counts (no reset: see whb). Per bucket k: incremented (h_ik: slot qw_r,
  // bucket wh_r, passes the read point) or decremented (h_dk: slot qw_f's write, bucket wh_f,
  // goes out), neither when both hit it; each a function of the late h_inc / h_dec and of the
  // bucket decodes, and an increment's value where the bucket is wh_r and h_inc is set (h_iv)
  logic [NH-1:0] h_ik [NCH], h_dk [NCH], h_iv [NCH];
  always_comb
    for (int c = 0; c < NCH; c++) begin
      h_inc[c] = (src[c] == S_WR && ctk[c]) || (qw_rn[c] != 0 && !wp_r[c]);
      h_dec[c] = src[c] == S_WW && wiss[c];
      for (int k = 0; k < NH; k++) begin
        h_ik[c][k] = h_inc[c] && wh_r[c] == HW'(k) && !(h_dec[c] && wh_f[c] == wh_r[c]);
        h_dk[c][k] = h_dec[c] && wh_f[c] == HW'(k) && !(h_inc[c] && wh_f[c] == wh_r[c]);
        h_iv[c][k] = h_inc[c] && wh_r[c] == HW'(k);
      end
    end
  // slot qw_r's beat, bucket and partial flag, registered: read from the LUT RAMs and wpart by
  // qw_r, they started the fill read's hazard check (w_blk) and so the command select and the
  // hazard count updates (the 5e5a build's nmem > nmem paths: qw_r -> whm -> wnz -> ... -> whb).
  // qw_r moves by one exactly on h_inc; the slot it then points to is the tail, whose write
  // (see g_mem) lands in it this cycle, when no entry past the read point is left (qw_rn 1, or 0
  // while it stays), otherwise the next slot as stored
  always_ff @(posedge clk)
    for (int c = 0; c < NCH; c++) begin
      logic tw;
      tw = !qw_n[c][WW] && qw_rn[c] == (h_inc[c] ? (WW + 1)'(1) : (WW + 1)'(0));
      if (tw) begin
        wadr_r[c] <= gb[c].m; wh_r[c] <= whash(gb[c].m); wp_r[c] <= SW_FILL && !(&gb[c].strb);
      end else if (h_inc[c]) begin
        wadr_r[c] <= wadr_n[c]; wh_r[c] <= wh_n[c]; wp_r[c] <= wpart[c][WW'(qw_r[c] + 1'b1)];
      end
    end
  always_ff @(posedge clk)
    for (int c = 0; c < NCH; c++)
      for (int k = 0; k < NH; k++)
        if (h_ik[c][k] || h_dk[c][k])
          whb[c][k] <= !h_iv[c][k] ? whb[c][wh_f[c]] - 1'b1 :
                       wnz[c][wh_r[c]] ? whb[c][wh_r[c]] + 1'b1 : (WW + 1)'(1);

`ifndef SYNTHESIS
  // The hazard counts are read only while live: a write goes out only from a live bucket, and
  // every live count is what a count reset with the rest would hold (whb_ref). The counts start
  // with anything, as after a reset in the middle of a run
  logic [WW:0] whb_ref [NCH][NH];
  initial for (int c = 0; c < NCH; c++) for (int k = 0; k < NH; k++) whb[c][k] = (WW + 1)'($urandom);
  always_ff @(posedge clk)
    if (rst) begin
      for (int c = 0; c < NCH; c++) for (int k = 0; k < NH; k++) whb_ref[c][k] <= '0;
    end else
      for (int c = 0; c < NCH; c++) begin
        if (h_dec[c] && !wnz[c][wh_f[c]])
          $fatal(1, "otpu_native_dram: channel %0d bucket %0d written out while not live", c, wh_f[c]);
        for (int k = 0; k < NH; k++)
          if (wnz[c][k] != (whb_ref[c][k] != 0) || (wnz[c][k] && whb[c][k] != whb_ref[c][k]))
            $fatal(1, "otpu_native_dram: channel %0d bucket %0d count %0d (live %0d), expected %0d",
                   c, k, whb[c][k], wnz[c][k], whb_ref[c][k]);
        if (h_inc[c] && !(h_dec[c] && wh_f[c] == wh_r[c]))
          whb_ref[c][wh_r[c]] <= whb_ref[c][wh_r[c]] + 1'b1;
        if (h_dec[c] && !(h_inc[c] && wh_f[c] == wh_r[c]))
          whb_ref[c][wh_f[c]] <= whb_ref[c][wh_f[c]] - 1'b1;
      end

  // b_par is the fold of the address. The A and B queues and the tag FIFO write their tails
  // every cycle: the head of each (the entry a command or a read's data takes) is what a queue
  // written on a push only holds (qar, qbr, btr: the old write enables)
  always_ff @(posedge clk)
    if (!rst && b_req && b_par != fold(b_addr))
      $fatal(1, "otpu_native_dram: b_par %0d, address %h", b_par, b_addr);
  always_ff @(posedge clk)
    if (!rst && w_req && (!HAS_W || (w_one ? !XPOST || w_addr[$clog2(D / 4)-1:0] != '0
                                           : w_addr[PWW-1:0] != '0)))
      $fatal(1, "otpu_native_dram: a port W request %s (address %h)",
             !HAS_W ? "without port W" : !w_one ? "off its wide chunk's start"
             : XPOST ? "of one beat off its chunk's start" : "of one beat without XPOST", w_addr);
  logic [$bits(qb_t)-1:0] qbr [NCH][QD];
  logic [$bits(qa_t)-1:0] qar [NCH][QD];
  bo_t btr [OD];
  always_ff @(posedge clk) begin
    for (int c = 0; c < NCH; c++) if (qb_push[c]) qbr[c][QW'(qb_h[c] + qb_n[c])] <= qb_e[c];
    for (int c = 0; c < NCH; c++) if (qa_push[c]) qar[c][QW'(qa_h[c] + qa_n[c])] <= qa_e[c];
    if (b_take && !b_we) btr[OW'(bt_h + bt_n)] <= '{k: b_k, h: b_h, tag: b_tag};
  end
  always_ff @(posedge clk)
    if (!rst) begin
      for (int c = 0; c < NCH; c++) begin
        qb_t r;
        r = qb_t'(qbr[c][QW'(qb_h[c])]);
        if (qb_n[c] != 0 && (WPOST ? {hb[c].we, hb[c].m} != {r.we, r.m} : hb[c] != r))
          $fatal(1, "otpu_native_dram: channel %0d B queue head (slot %0d) differs", c, qb_h[c]);
        if (qa_n[c] != 0 && ha[c] != qa_t'(qar[c][QW'(qa_h[c])]))
          $fatal(1, "otpu_native_dram: channel %0d A queue head (slot %0d) differs", c, qa_h[c]);
      end
      if (bt_n != 0 && bth != btr[bt_h])
        $fatal(1, "otpu_native_dram: B tag %0d at %0d, expected %0d", bth, bt_h, btr[bt_h]);
      if (bt_n >= 2 && bt2 != btr[OW'(bt_h + 1'b1)])
        $fatal(1, "otpu_native_dram: B tag %0d at %0d + 1, expected %0d", bt2, bt_h, btr[OW'(bt_h + 1'b1)]);
    end

  // the B FIFO's read ahead named the head: rb_hn was rb_h's next value, and the SW queue's:
  // qw_fn was qw_f's next value; (RBB 0) every copy of the B FIFO's tail is the tail
  bit rst_seen;                                 // (registers start arbitrary)
  logic [REW-1:0] rb_hq [NCH];
  logic [WW-1:0] qw_fq [NCH];
  initial rst_seen = 1'b0;
  always_ff @(posedge clk) begin
    if (rst) rst_seen <= 1'b1;
    rb_hq <= rb_hn;
    qw_fq <= qw_fn;
    if (!rst && rst_seen)
      for (int c = 0; c < NCH; c++) begin
        if (rb_hq[c] != rb_h[c])
          $fatal(1, "otpu_native_dram: channel %0d B head %0d, its read ahead %0d", c, rb_h[c],
                 rb_hq[c]);
        if (qw_fq[c] != qw_f[c])
          $fatal(1, "otpu_native_dram: channel %0d SW head %0d, its read ahead %0d", c, qw_f[c],
                 qw_fq[c]);
        for (int k = 0; k < RBP; k++)
          if (!RBB && rb_tq[c][k] != rb_t[c])
            $fatal(1, "otpu_native_dram: channel %0d B tail copy %0d is %0d, the tail %0d", c, k,
                   rb_tq[c][k], rb_t[c]);
      end
  end

  // gb and, a cycle later, gd are, while valid, what a buffer merging each write as it is taken
  // holds (gb_ref; gd in the bytes written)
  qw_t gb_ref [NCH];
  logic [CB*8-1:0] gd_ref [NCH];
  logic [CB-1:0] gs_ref [NCH];
  logic [NCH-1:0] gd_rv;
  function automatic logic [CB*8-1:0] written(input logic [CB*8-1:0] d, input logic [CB-1:0] strb);
    for (int k = 0; k < CB; k++) if (!strb[k]) d[8 * k +: 8] = 8'h00;
    return d;
  endfunction
  always_ff @(posedge clk)
    for (int c = 0; c < NCH; c++) begin
      if (!rst && gv[c] && (gb[c].m != gb_ref[c].m || gb[c].strb != gb_ref[c].strb))
        $fatal(1, "otpu_native_dram: channel %0d SW gather buffer %h differs from %h", c, gb[c].m,
               gb_ref[c].m);
      if (!rst && gd_rv[c] && written(gd[c], gs_ref[c]) != gd_ref[c])
        $fatal(1, "otpu_native_dram: channel %0d SW gather data differs", c);
      gd_ref[c] <= gb_ref[c].data;
      gs_ref[c] <= gb_ref[c].strb;
      gd_rv[c] <= gv[c];
      if (sw_take && sw_ch == CL'(c)) begin
        logic same;
        same = gv[c] && gb_ref[c].m == sw_m;
        gb_ref[c].m <= sw_m;
        for (int k = 0; k < CB; k++) begin
          logic hit;
          hit = (sw_addr[WB-1:0] >> SSL) == WB'(k / SB) && sw_be[k % SB];
          gb_ref[c].data[8 * k +: 8] <= hit ? sw_wdata[8 * (k % SB) +: 8]
                                        : same ? gb_ref[c].data[8 * k +: 8] : 8'h00;
          gb_ref[c].strb[k] <= hit || (same && gb_ref[c].strb[k]);
        end
      end
    end

  // The SW queue's slots are written at the tail every cycle as well: its slots at the read point
  // (qw_r) and at the head (qw_f) are what a queue written on a push only holds (the head's beat,
  // once past the read point: see u_qwd); the read point's registers (wadr_r, wh_r, wp_r) are what
  // its slot holds
  logic [$bits(qw_t)-1:0] qwr [NCH][WQD];
  logic [CAW-1:0] war [NCH][WQD];
  logic [WQD-1:0] wpr [NCH], wgr [NCH];
  always_ff @(posedge clk)
    for (int c = 0; c < NCH; c++) begin
      if (qw_push[c]) begin
        qwr[c][WW'(qw_f[c] + qw_n[c])] <= gb_ref[c];
        war[c][WW'(qw_f[c] + qw_n[c])] <= gb[c].m;
        wpr[c][WW'(qw_f[c] + qw_n[c])] <= SW_FILL && !(&gb[c].strb);
        wgr[c][WW'(qw_f[c] + qw_n[c])] <= 1'b0;
      end
      if (n_rvalid[c] && rl[c] && tgh[c].k == K_W) wgr[c][tgh[c].slot] <= 1'b1;
    end
  always_ff @(posedge clk)
    if (!rst)
      for (int c = 0; c < NCH; c++) begin
        qw_t hr;
        if ((src[c] == S_WR && ctk[c]) && (qw_rn[c] != 0 && !wp_r[c]))
          $fatal(1, "otpu_native_dram: channel %0d fill read of a whole beat", c);
        if (qw_rn[c] != 0 && (wadr_r[c] != wadr_o[c] || wh_r[c] != wh_o[c] ||
                              wp_r[c] != wpart[c][qw_r[c]]))
          $fatal(1, "otpu_native_dram: channel %0d read point slot %0d: registered %h %0d %0d, stored %h %0d %0d",
                 c, qw_r[c], wadr_r[c], wh_r[c], wp_r[c], wadr_o[c], wh_o[c], wpart[c][qw_r[c]]);
        if (qw_rn[c] != 0 && (wadr_r[c] != war[c][qw_r[c]] || wh_r[c] != whash(war[c][qw_r[c]]) ||
                              wp_r[c] != wpr[c][qw_r[c]]))
          $fatal(1, "otpu_native_dram: channel %0d SW slot %0d at the read point differs", c, qw_r[c]);
        if (qw_n[c] != 0 && (wh_f[c] != whash(war[c][qw_f[c]]) || wpart[c][qw_f[c]] != wpr[c][qw_f[c]] ||
                             wgot[c][qw_f[c]] != wgr[c][qw_f[c]]))
          $fatal(1, "otpu_native_dram: channel %0d SW head slot %0d differs", c, qw_f[c]);
        hr = qw_t'(qwr[c][qw_f[c]]);
        if (qw_n[c] != qw_rn[c] && (WPOST ? hw[c].m != hr.m
                                    : qw_t'{m: hw[c].m, data: written(hw[c].data, hw[c].strb),
                                            strb: hw[c].strb} != hr))
          $fatal(1, "otpu_native_dram: channel %0d SW head slot %0d's beat differs", c, qw_f[c]);
      end

  // WPOST: each write that goes out on a channel is the oldest one there whose beat went in and
  // whose write has not gone out (its source and channel beat; the beat itself is what the
  // gather or the take held: the memory checks it), never into a full FIFO; the stage waits
  // exactly when the SW beat goes in on its channel (SWG: on any channel; and the beats in a
  // cycle are one at most); port A carries no writes
  typedef struct { logic b; logic [CAW-1:0] m; } wp_t;
  wp_t wpq [NCH][$];
  logic [CAW-1:0] bsa;                          // the staged beat's channel beat, the SW beat's
  logic [CAW-1:0] spa [NCH];
  always_ff @(posedge clk) begin
    if (!bh) bsa <= b_addr[PWW +: CAW];
    for (int c = 0; c < NCH; c++) spa[c] <= gb[c].m;
  end
  always_ff @(posedge clk)
    if (rst) begin
      for (int c = 0; c < NCH; c++) wpq[c].delete();
    end else if (WPOST) begin
`ifndef OTPU_WO_UNCHECKED                 // (the mutants' second run: the memory must catch it)
      if (a_take && a_we) $fatal(1, "otpu_native_dram: a port A write with WPOST");
      if (bh != (bsv && (SWG ? |sp || whv : sp[bsc])))
        $fatal(1, "otpu_native_dram: the B stage waits (%0d) off its channel's SW beat", bh);
      if (whv && gi) $fatal(1, "otpu_native_dram: WHB: a beat goes in under a high half");
      if (SWG && $countones(n_wvalid) > 1)
        $fatal(1, "otpu_native_dram: SWG: %0d beats posted in a cycle", $countones(n_wvalid));
      for (int c = 0; c < NCH; c++) begin
        // the write order (registers) holds the beats in whose writes have not gone out
        if (32'(won[c]) != wpq[c].size() || won[c] > (WOW + 1)'(WOD) ||
            (won[c] != 0 && (wo1[c] != wov[c][woh[c]] || wo1[c] != wpq[c][0].b)))
          $fatal(1, "otpu_native_dram: channel %0d write order: %0d entries (head %0d), %0d beats in",
                 c, won[c], wo1[c], wpq[c].size());
        if (n_wvalid[c] && !n_wready[c])
          $fatal(1, "otpu_native_dram: channel %0d: a beat posted into a full write-data FIFO", c);
        if (sp[c] || bst[c]) wpq[c].push_back('{b: !sp[c], m: sp[c] ? spa[c] : bsa});
        if (wiss[c]) begin
          if (wpq[c].size() == 0)
            $fatal(1, "otpu_native_dram: channel %0d: a write out with no beat in", c);
          else if (wpq[c][0].b != (src[c] == S_B) || wpq[c][0].m != n_caddr[c])
            $fatal(1, "otpu_native_dram: channel %0d: a %s write to beat %h out, the next beat in is %s's to %h",
                   c, src[c] == S_B ? "B" : "SW", n_caddr[c], wpq[c][0].b ? "B" : "SW", wpq[c][0].m);
          else void'(wpq[c].pop_front());
        end
      end
`endif
    end

  // Read data never arrives without a read in flight or without room reserved for it (the
  // native port has no backpressure on read data)
  always_ff @(posedge clk)
    if (!rst)
      for (int c = 0; c < NCH; c++) begin
        if (n_rvalid[c] && tg_n[c] == 0)
          $fatal(1, "otpu_native_dram: channel %0d read data with no read in flight", c);
        if (n_rvalid[c] && tg_n[c] != 0 && tgh[c].k == K_B && rb_n[c] == (REW + 1)'(RE))
          $fatal(1, "otpu_native_dram: channel %0d B read FIFO overflow", c);
        if (n_rvalid[c] && tg_n[c] != 0 && tgh[c].k == K_A && ra_n[c] == (AEW + 1)'(AE))
          $fatal(1, "otpu_native_dram: channel %0d A read FIFO overflow", c);
        if (n_rvalid[c] && tg_n[c] != 0 && tgh[c].k == K_X && xi[c] == 0)
          $fatal(1, "otpu_native_dram: channel %0d W read data with no W read in flight", c);
`ifndef OTPU_WO_UNCHECKED
        if (XPOST && PCW'(p_cnt[c] - pc_q[c]) > PCW'(xi[c]))
          $fatal(1, "otpu_native_dram: channel %0d posted %0d W beats, %0d W reads in flight", c,
                 PCW'(p_cnt[c] - pc_q[c]), xi[c]);
`endif
        if (n_rvalid[c] && tg_n[c] != 0 && tgh[c].k == K_W && wgot[c][tgh[c].slot])
          $fatal(1, "otpu_native_dram: channel %0d SW fill data for a filled slot", c);
        if (rtk[c] && tg_n[c] == (TW + 1)'(TD) && !(n_rvalid[c] && rl[c]))
          $fatal(1, "otpu_native_dram: channel %0d read tag FIFO overflow", c);
      end

  // Counters for the simulation's statistics (otpu_top prints them): per channel, A runs and
  // their read commands, B reads, SW fill reads, W reads; writes by source, and those with a byte
  // mask that is not whole (a read-modify-write in the channel module)
  longint st_arun [NCH], st_ard [NCH], st_brd [NCH], st_srd [NCH], st_xrd [NCH];
  longint st_bwr [NCH], st_awr [NCH], st_swr [NCH], st_pb [NCH], st_pa [NCH], st_ps [NCH];
  longint st_cr = 0, st_bh = 0;                 // WPOST: cycles B and SW wait on room, on a B beat
  initial
    for (int c = 0; c < NCH; c++) begin
      st_arun[c] = 0; st_ard[c] = 0; st_brd[c] = 0; st_srd[c] = 0; st_xrd[c] = 0; st_bwr[c] = 0;
      st_awr[c] = 0; st_swr[c] = 0; st_pb[c] = 0; st_pa[c] = 0; st_ps[c] = 0;
    end
  always_ff @(posedge clk) begin
    if (WPOST && !rst && !(&ok)) st_cr <= st_cr + 1;
    if (WPOST && !rst && bh) st_bh <= st_bh + 1;
  end
  always_ff @(posedge clk)
    for (int c = 0; c < NCH; c++) begin
      if (rtk[c] && src[c] == S_A) begin
        st_ard[c] <= st_ard[c] + 1;
        if (a_iss[c] == 0) st_arun[c] <= st_arun[c] + 1;
      end
      if (rtk[c] && src[c] == S_B) st_brd[c] <= st_brd[c] + 1;
      if (rtk[c] && src[c] == S_WR) st_srd[c] <= st_srd[c] + 1;
      if (ctk[c] && src[c] == S_X) st_xrd[c] <= st_xrd[c] + 1;
      if (wiss[c]) begin
        logic part;
        part = !WPOST && n_wmask[c] != '1;    // (WPOST: as its beat goes in, below)
        case (src[c])
          S_A: begin st_awr[c] <= st_awr[c] + 1; if (part) st_pa[c] <= st_pa[c] + 1; end
          S_WW: begin st_swr[c] <= st_swr[c] + 1; if (part) st_ps[c] <= st_ps[c] + 1; end
          default: begin st_bwr[c] <= st_bwr[c] + 1; if (part) st_pb[c] <= st_pb[c] + 1; end
        endcase
      end
      if (WPOST && (sp[c] || bst[c]) && (WHB ? wpm != '1 : n_wmask[c] != '1)) begin
        if (sp[c]) st_ps[c] <= st_ps[c] + 1;
        else st_pb[c] <= st_pb[c] + 1;
      end
    end
`endif
endmodule
