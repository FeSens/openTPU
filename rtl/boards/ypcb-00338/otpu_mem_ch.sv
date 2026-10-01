// One DDR3 channel's user side: the accelerator's native master (otpu_native_dram, core clock,
// one 64-byte beat per command) and XDMA's AXI master (host loads and reads, axi_aclk, 128-bit)
// onto the two native ports of the channel's controller (uclk: LiteDRAM's sys). With
// otpu_axi_split2 it replaced the MIG builds' SmartConnect and the MIG's AXI front end.
//
// Each master's commands cross into uclk through its own asynchronous FIFOs (commands, write
// data) and its read data comes back through another; an arbiter in uclk takes up to RUN commands
// from one master before it lets the other in, one command per cycle, and each master's commands
// in its order. The accelerator's write data comes in write-command order, before or after its
// command: a write goes when both are at the heads of their FIFOs. The controller returns a port's
// read data in that port's command order and cannot hold it back, so one tag per read (its master,
// its slot) routes the data back, and a master's read is issued only when its read-data FIFO has
// room for it and for every read not yet through (credits). The accelerator's read data leaves
// without backpressure too (its master reserves room for its reads), from a register.
//
// Two ports: LiteDRAM's crossbar lets a master have commands in one bank at a time (the next
// bank's command waits until the previous bank's last column command), which idles the channel
// at every bank change (docs/litedram.md section 11). So each command goes to the port of its
// beat's bank parity (beat bit 7: ROW_BANK_COLUMN, 8 KiB rows), through that port's own output
// command queue (OQ entries, then a register), write-data FIFO and read tags, and a port whose
// bank waits holds back only its own commands. A beat is always on the same port and each port
// keeps its commands in order, so the ordering below holds per beat; the two ports' commands are
// not ordered against each other. They share the channel's data bus: read data comes back from
// one port at a time, each port's in its order. Each read takes the next slot of its master's
// read-data FIFO when it issues (the credits keep it free); its data is written there when its
// port returns it, and the FIFO lets it through once every earlier slot is written (otpu_afifo
// OOO), so a master sees its read data in command order.
//
// The controller port has LiteDRAM's native-port semantics: commands (we, beat), write data taken
// (wdata ready) some cycles after its command, in write-command order, and read data in command
// order. LiteDRAM's crossbar takes the write data when the bank asks for it without looking at
// wdata valid, so a write's data enters its port's output data FIFO (only the controller's reset
// clears it) in the cycle its command enters the port's output command queue (the arbiter sees
// only registered state): the data is there before the controller can take the command, and the
// per-master holds cannot take either back. (LiteDRAMNativePortECC puts a one-beat register
// in front of the crossbar that takes the head as soon as it is empty; it is loaded well before
// the bank asks, which is at least 4 cycles after the command. Its rdata ready must be tied high.)
//
// Partial beats (any byte not written): LiteDRAM's ECC port rejects partial writes, and the board
// has no DM pins, so the read-modify-write is done here. The arbiter issues a read of the beat and
// stops; the write's command and data stay at the heads of their master's FIFOs; when the read's
// data comes back (its tag), the old bytes fill the lanes not written and the whole beat is
// written, and only then does the arbiter go on. So nothing reaches the controller between the
// read and the write: the read comes after every command issued before it, the write before every
// command issued after it (both on the beat's port), and the controller keeps one port's commands
// in order (LiteDRAM locks a port to one bank until that bank's queue has drained; each bank
// machine is a FIFO). A partial beat costs about one read latency of the whole channel.
//
// n_wdone counts the accelerator's write beats the controller has taken on either port (the
// command handshake; the data is then already in the output FIFO), gray-coded across into the
// core clock. Every command to the same beat issued after that, from either master, reaches the
// controller behind the write (on the same port), so the write is visible to both masters. An
// XDMA write burst gets its B the same way, once all of its 64-byte beats are taken. XDMA's 16-byte
// beats are packed into 64-byte beats per burst (a partial beat at either end is a partial write).
//
// Addresses: n_caddr and bits 30:6 of XDMA's byte address select the 64-byte beat in the channel
// (2 GiB; XDMA's bit 31, the channel, is dropped by otpu_axi_split2). XDMA's bursts are INCR of
// 16-byte beats (AxSIZE 4) and may start anywhere in a 64-byte beat; narrow bursts are not
// supported (XDMA issues none). Its reads return in order (legal for any IDs); responses are OKAY.
//
// n_err: the controller broke its side of the port contract -- read data from both ports in one
// cycle, or on a port with no read outstanding. Sticky until the controller's reset, in the core
// clock (otpu_board: STATUS AXI_ERR), so the card shows it too; the simulation stops on it.
//
// Resets: each master's path has a hold in uclk -- the controller's reset, or the master's own
// reset (synchronized, and kept up until the hold is seen in the master's clock, so that a reset
// of a cycle gets a whole hold too), then 16 cycles, and on until its reads in flight have come
// back (their data is dropped), a read-modify-write of its own is done and no command of its own
// is left in the output command queues (a write counted after the hold would count one the
// master issued before its reset: its write count restarts from n_wdone's reset value, 0) -- and
// the same hold, synchronized, in the master's clock. The clock-crossing FIFOs reset only on the
// hold, in both clocks, so neither side ever sees the other's pointers reset under it; the
// master's side of the bridge resets on its reset or the hold. What the arbiter has issued is
// finished regardless (a merged write whose master is held is dropped: its data went with the
// FIFO). So a PCIe link reset (XDMA's axi_aresetn) while the accelerator runs costs neither the
// other master nor the memory.
//
// Clock-domain crossings: every synchronizer carries ASYNC_REG; their constraints (max delay
// without skew, bus skew for the gray counts) are in boards/ypcb-00338/constraints/otpu_mem_ch.tcl.
module otpu_mem_ch #(
  parameter int XIDW = 4,         // XDMA M_AXI ID width
  parameter int ARD  = 64,        // accelerator read-data FIFO (read credits), beats
  parameter int XRD  = 16,        // XDMA read-data FIFO, beats
  parameter int RUN  = 32,        // commands from one master before the arbiter may switch
  parameter int OQ   = 32,        // output command queue per port (and one register)
  parameter int OD   = 32         // output write-data FIFO per port: writes issued, data not taken
) (
  // accelerator (otpu_native_dram), core clock
  input  logic            clk,
  input  logic            rst,
  input  logic            n_cvalid,
  output logic            n_cready,
  input  logic            n_cwe,
  input  logic [24:0]     n_caddr,    // 64-byte beat in the channel
  input  logic            n_wvalid,   // one beat per write command, in write-command order
  output logic            n_wready,
  input  logic [511:0]    n_wdata,
  input  logic [63:0]     n_wmask,    // 1 = write the byte
  output logic            n_rvalid,   // in read-command order, no backpressure
  output logic [511:0]    n_rdata,
  output logic [15:0]     n_wdone,    // write beats taken by the controller, mod 2^16
  output logic            n_err,      // the controller broke the port contract (sticky)
  // XDMA, axi_aclk
  input  logic            xclk,
  input  logic            xrst,
  input  logic            x_awvalid,
  output logic            x_awready,
  input  logic [XIDW-1:0] x_awid,
  input  logic [31:0]     x_awaddr,
  input  logic [7:0]      x_awlen,
  input  logic            x_wvalid,
  output logic            x_wready,
  input  logic [127:0]    x_wdata,
  input  logic [15:0]     x_wstrb,
  input  logic            x_wlast,
  output logic            x_bvalid,
  input  logic            x_bready,
  output logic [XIDW-1:0] x_bid,
  output logic [1:0]      x_bresp,
  input  logic            x_arvalid,
  output logic            x_arready,
  input  logic [XIDW-1:0] x_arid,
  input  logic [31:0]     x_araddr,
  input  logic [7:0]      x_arlen,
  output logic            x_rvalid,
  input  logic            x_rready,
  output logic [XIDW-1:0] x_rid,
  output logic [127:0]    x_rdata,
  output logic [1:0]      x_rresp,
  output logic            x_rlast,
  // the controller's two native ports, uclk: port p takes the beats of the banks of parity p
  input  logic            uclk,
  input  logic            urst,
  output logic [1:0]      c_cmd_valid,
  input  logic [1:0]      c_cmd_ready,
  output logic [1:0]      c_cmd_we,
  output logic [1:0][24:0] c_cmd_addr,
  output logic [1:0]      c_wdata_valid,
  input  logic [1:0]      c_wdata_ready,
  output logic [1:0][511:0] c_wdata_data,
  output logic [1:0][63:0] c_wdata_we,    // 1 = write the byte (all ones: whole beats)
  input  logic [1:0]      c_rdata_valid,  // each port's in its command order, no backpressure;
                                          //   one port at a time
  input  logic [1:0][511:0] c_rdata_data  // one bus: port 1's is port 0's (only port 0's is read)
);
  localparam int QW = 26;                  // command: {write, beat[24:0]}
  localparam int DW = 577;                 // write data: {partial, byte enables[63:0], data[511:0]}
  localparam int CW = 16;                  // write-accept counters (beats, modulo 2^16)
  localparam int AOW = $clog2(ARD) + 1;
  localparam int XOW = $clog2(XRD) + 1;
  localparam int ASW = $clog2(ARD);        // read slots
  localparam int XSW = $clog2(XRD);
  localparam int TW  = 2 + ASW;            // read tag: {read-modify-write, xdma, slot}
  initial if (ARD + XRD + 1 > 128) $fatal(1, "otpu_mem_ch: ARD + XRD reads in flight exceed the tag FIFO");
  initial if (XRD > ARD) $fatal(1, "otpu_mem_ch: XRD > ARD (the tag's slot)");

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
  // The controller's reset (urst: the core's sys reset register, which drives both channels'
  // bridges and the core's own loads) registered once here (keep), by this channel's bridge; the
  // bridge's controller side takes it (urq), a cycle after the controller. In the fused c2830d6
  // build at 133.33 MHz, crg_rst1 -> u_ch0 g_port[*].u_oq wp / rp R (fanout 500, 0 levels,
  // 6.6 ns of route, +0.059) was the worst path. The cycle in which the controller is out of
  // reset and the bridge not: every master is held (the holds last 15 cycles past urq), so no
  // command enters the queues, and the controller, reset with no reads, returns no data (checked
  // below); on the way in, the bridge runs a cycle into the controller's reset, and what it
  // queues then goes with its own reset a cycle later.
  (* keep = "true" *) logic urq = 1'b1;
  always_ff @(posedge uclk) urq <= urst;
  (* ASYNC_REG = "TRUE" *) logic a_rs1, a_rs2, x_rs1, x_rs2;    // master resets in uclk
  (* ASYNC_REG = "TRUE" *) logic a_hs1 = 1'b1, a_hs2 = 1'b1;     // holds in the masters' clocks
  (* ASYNC_REG = "TRUE" *) logic x_hs1 = 1'b1, x_hs2 = 1'b1;
  logic a_hold = 1'b1, x_hold = 1'b1;
  logic [3:0] a_hcnt, x_hcnt;
  logic a_crst = 1'b1, x_crst = 1'b1;      // the masters' sides of the bridge
  logic a_req = 1'b0, x_req = 1'b0;        // a master's reset, held until its hold is seen back
  logic [AOW-1:0] a_out;                   // reads in flight (issued, data not yet in the FIFO)
  logic [XOW-1:0] x_out;
  logic rm_busy, rm_x;                     // a read-modify-write in progress, and its master
  logic a_oc, x_oc;                        // a command of the master's in the output queues
  always_ff @(posedge uclk) begin
    a_rs1 <= a_req; a_rs2 <= a_rs1;
    x_rs1 <= x_req; x_rs2 <= x_rs1;
    a_hcnt <= (urq || a_rs2) ? 4'hF : (a_hcnt != 0) ? a_hcnt - 1'b1 : a_hcnt;
    x_hcnt <= (urq || x_rs2) ? 4'hF : (x_hcnt != 0) ? x_hcnt - 1'b1 : x_hcnt;
    a_hold <= urq || a_rs2 ||
              (a_hold && (a_hcnt != 0 || a_out != 0 || (rm_busy && !rm_x) || a_oc));
    x_hold <= urq || x_rs2 ||
              (x_hold && (x_hcnt != 0 || x_out != 0 || (rm_busy && rm_x) || x_oc));
  end
  // A reset shorter than the round trip through the synchronizers would otherwise release the
  // master's side before its hold arrives, with the old read data still in its FIFO: the request
  // stays up until the hold is seen here, and the side stays in reset until the hold is over.
  always_ff @(posedge clk) begin
    a_req <= rst || (a_req && !a_hs2);
    a_hs1 <= a_hold; a_hs2 <= a_hs1;
    a_crst <= rst || a_req || a_hs2;
  end
  always_ff @(posedge xclk) begin
    x_req <= xrst || (x_req && !x_hs2);
    x_hs1 <= x_hold; x_hs2 <= x_hs1;
    x_crst <= xrst || x_req || x_hs2;
  end

  // ================================================================ clock crossings
  logic          aq_wv, aq_wr, aq_rv, aq_rr;  logic [QW-1:0] aq_wd, aq_rd;
  logic          ad_wv, ad_wr, ad_rv, ad_rr;  logic [DW-1:0] ad_wd, ad_rd;
  logic          ar_wv, ar_wr, ar_rv, ar_rr;  logic [511:0]  ar_wd, ar_rd;
  logic          xq_wv, xq_wr, xq_rv, xq_rr;  logic [QW-1:0] xq_wd, xq_rd;
  logic          xd_wv, xd_wr, xd_rv, xd_rr;  logic [DW-1:0] xd_wd, xd_rd;
  logic          xr_wv, xr_wr, xr_rv, xr_rr;  logic [511:0]  xr_wd, xr_rd;
  logic [AOW-1:0] ar_used;
  logic [XOW-1:0] xr_used;
  logic [5:0] unused_aq, unused_ad;
  logic [4:0] unused_xq, unused_xd;
  logic [$clog2(ARD)-1:0] ar_slot;         // the read-data FIFOs' slots (out-of-order writes)
  logic [$clog2(XRD)-1:0] xr_slot;
  logic ar_cmt, xr_cmt;

  otpu_afifo #(.W(QW), .DEPTH(32)) u_aq (.wclk(clk), .wrst(a_hs2), .wvalid(aq_wv), .wready(aq_wr),
    .wdata(aq_wd), .wslot('0), .wcommit(), .wused(unused_aq), .rclk(uclk), .rrst(a_hold),
    .rvalid(aq_rv), .rready(aq_rr), .rdata(aq_rd));
  otpu_afifo #(.W(DW), .DEPTH(32)) u_ad (.wclk(clk), .wrst(a_hs2), .wvalid(ad_wv), .wready(ad_wr),
    .wdata(ad_wd), .wslot('0), .wcommit(), .wused(unused_ad), .rclk(uclk), .rrst(a_hold),
    .rvalid(ad_rv), .rready(ad_rr), .rdata(ad_rd));
  // the read data, written out of order at each read's slot (the two ports return independently)
  otpu_afifo #(.W(512), .DEPTH(ARD), .OOO(1'b1)) u_ar (.wclk(uclk), .wrst(a_hold), .wvalid(ar_wv),
    .wready(ar_wr), .wdata(ar_wd), .wslot(ar_slot), .wcommit(ar_cmt), .wused(ar_used), .rclk(clk),
    .rrst(a_hs2), .rvalid(ar_rv), .rready(ar_rr), .rdata(ar_rd));
  // the XDMA side's write enables start from flip-flops (RWR: a registered wready)
  otpu_afifo #(.W(QW), .DEPTH(16), .RWR(1'b1)) u_xq (.wclk(xclk), .wrst(x_hs2), .wvalid(xq_wv),
    .wready(xq_wr), .wdata(xq_wd), .wslot('0), .wcommit(), .wused(unused_xq), .rclk(uclk),
    .rrst(x_hold), .rvalid(xq_rv), .rready(xq_rr), .rdata(xq_rd));
  otpu_afifo #(.W(DW), .DEPTH(16), .RWR(1'b1)) u_xd (.wclk(xclk), .wrst(x_hs2), .wvalid(xd_wv),
    .wready(xd_wr), .wdata(xd_wd), .wslot('0), .wcommit(), .wused(unused_xd), .rclk(uclk),
    .rrst(x_hold), .rvalid(xd_rv), .rready(xd_rr), .rdata(xd_rd));
  otpu_afifo #(.W(512), .DEPTH(XRD), .OOO(1'b1)) u_xr (.wclk(uclk), .wrst(x_hold), .wvalid(xr_wv),
    .wready(xr_wr), .wdata(xr_wd), .wslot(xr_slot), .wcommit(xr_cmt), .wused(xr_used), .rclk(xclk),
    .rrst(x_hs2), .rvalid(xr_rv), .rready(xr_rr), .rdata(xr_rd));

  // write beats taken by the controller, per master (gray counters from uclk)
  logic [CW-1:0] a_wacc, a_wacc_g, x_wacc, x_wacc_g;
  (* ASYNC_REG = "TRUE" *) logic [CW-1:0] a_wacc_s1, a_wacc_s2;
  (* ASYNC_REG = "TRUE" *) logic [CW-1:0] x_wacc_s1, x_wacc_s2;

  // ================================================================ accelerator side (clk)
  assign n_cready = !a_crst && aq_wr;
  assign aq_wv    = n_cvalid && n_cready;
  assign aq_wd    = {n_cwe, n_caddr};
  assign n_wready = !a_crst && ad_wr;
  assign ad_wv    = n_wvalid && n_wready;
  assign ad_wd    = {~&n_wmask, n_wmask, n_wdata};
  assign ar_rr    = 1'b1;
  always_ff @(posedge clk) begin
    n_rvalid <= ar_rv && !a_crst;
    n_rdata  <= ar_rd;
  end
  always_ff @(posedge clk) begin
    if (a_crst) begin a_wacc_s1 <= '0; a_wacc_s2 <= '0; n_wdone <= '0; end
    else begin a_wacc_s1 <= a_wacc_g; a_wacc_s2 <= a_wacc_s1; n_wdone <= g2b(a_wacc_s2); end
  end

  // ================================================================ XDMA side (xclk)
  // Read bursts: one command per 64-byte beat touched; the R channel hands out the 16-byte lanes
  // from the burst's first lane. Write bursts: 16-byte beats packed into 64-byte beats.
  localparam int XRQ = XIDW + 10;
  logic       xai_wv, xai_wr, xai_rv, xai_rr;  logic [31:0] xai_rd;      // {n - 1, beat}
  logic       xwi_wv, xwi_wr, xwi_rv, xwi_rr;  logic [26:0] xwi_rd;      // {lane, beat}
  logic       xrq_wv, xrq_wr, xrq_rv, xrq_rr;  logic [XRQ-1:0] xrq_rd;   // {id, len, lane}
  logic       xbq_wv, xbq_wr, xbq_rv, xbq_rr;  logic [XIDW+6:0] xbq_rd;  // {id, n - 1}
  logic [6:0] x_arn, x_awn;                // 64-byte beats of the burst, minus one
  assign x_arn = 7'(({1'b0, x_arlen} + {7'b0, x_araddr[5:4]}) >> 2);
  assign x_awn = 7'(({1'b0, x_awlen} + {7'b0, x_awaddr[5:4]}) >> 2);
  // their heads from registers (RO): the R, B and write-packing logic starts from flip-flops
  otpu_sfifo #(.W(32), .DEPTH(4), .RO(1'b1)) u_xai (.clk(xclk), .rst(x_crst), .wvalid(xai_wv),
    .wready(xai_wr), .wdata({x_arn, x_araddr[30:6]}), .rvalid(xai_rv), .rready(xai_rr),
    .rdata(xai_rd));
  otpu_sfifo #(.W(27), .DEPTH(4), .RO(1'b1)) u_xwi (.clk(xclk), .rst(x_crst), .wvalid(xwi_wv),
    .wready(xwi_wr), .wdata({x_awaddr[5:4], x_awaddr[30:6]}), .rvalid(xwi_rv), .rready(xwi_rr),
    .rdata(xwi_rd));
  otpu_sfifo #(.W(XRQ), .DEPTH(16), .RO(1'b1)) u_xrq (.clk(xclk), .rst(x_crst), .wvalid(xrq_wv),
    .wready(xrq_wr), .wdata({x_arid, x_arlen, x_araddr[5:4]}), .rvalid(xrq_rv), .rready(xrq_rr),
    .rdata(xrq_rd));
  otpu_sfifo #(.W(XIDW + 7), .DEPTH(16), .RO(1'b1)) u_xbq (.clk(xclk), .rst(x_crst),
    .wvalid(xbq_wv), .wready(xbq_wr), .wdata({x_awid, x_awn}), .rvalid(xbq_rv), .rready(xbq_rr),
    .rdata(xbq_rd));
  assign x_arready = !x_crst && xai_wr && xrq_wr;
  assign xai_wv    = x_arvalid && x_arready;
  assign xrq_wv    = xai_wv;
  assign x_awready = !x_crst && xwi_wr && xbq_wr;
  assign xwi_wv    = x_awvalid && x_awready;
  assign xbq_wv    = xwi_wv;

  // write packing: the burst's current 64-byte beat collects its lanes; it goes out with its
  // last lane or the burst's last beat. A completed write beat goes first, else a read beat.
  logic [6:0]   x_roff, x_woff;
  logic [1:0]   x_wlane_r, x_wlane;
  logic         x_wfirst, x_wpush, x_selw, x_rlastb;
  logic [511:0] x_wbuf, x_wdata_full;
  logic [63:0]  x_wstb, x_wstb_full;
  logic [3:0]   x_wsfull;                // the lanes of x_wstb with every strobe set
  logic         x_wfull;                 // x_wstb_full all set (no AND over its 64 bits)
  assign x_wlane  = x_wfirst ? xwi_rd[26:25] : x_wlane_r;
  assign x_wpush  = x_wlane == 2'd3 || x_wlast;
  assign x_selw   = xwi_rv && x_wvalid && x_wpush;
  assign x_rlastb = x_roff == xai_rd[31:25];
  always_comb begin
    x_wdata_full = x_wbuf;
    x_wstb_full  = x_wstb;
    x_wdata_full[x_wlane * 128 +: 128] = x_wdata;
    x_wstb_full[x_wlane * 16 +: 16]    = x_wstrb;
  end
  assign xq_wv    = !x_crst && xq_wr && (x_selw ? xd_wr : xai_rv);
  assign xq_wd    = x_selw ? {1'b1, 25'(xwi_rd[24:0] + x_woff)} : {1'b0, 25'(xai_rd[24:0] + x_roff)};
  assign x_wready = !x_crst && xwi_rv && (!x_wpush || (xq_wr && xd_wr));
  assign xd_wv    = x_wvalid && x_wready && x_wpush;
  assign x_wfull  = &x_wstrb && &(x_wsfull | (4'b1 << x_wlane));
  assign xd_wd    = {~x_wfull, x_wstb_full, x_wdata_full};
  assign xai_rr   = xq_wv && !x_selw && x_rlastb;
  assign xwi_rr   = x_wvalid && x_wready && x_wlast;
  always_ff @(posedge xclk) begin
    if (x_crst) begin
      x_roff <= '0; x_woff <= '0; x_wlane_r <= '0; x_wfirst <= 1'b1; x_wbuf <= '0; x_wstb <= '0;
      x_wsfull <= '0;
    end else begin
      if (xq_wv && !x_selw) x_roff <= x_rlastb ? '0 : x_roff + 1'b1;
      if (x_wvalid && x_wready) begin
        x_wfirst  <= x_wlast;
        x_wlane_r <= x_wlane + 1'b1;
        if (x_wpush) begin
          x_woff <= x_wlast ? '0 : x_woff + 1'b1;
          x_wbuf <= '0; x_wstb <= '0; x_wsfull <= '0;
        end else begin
          x_wbuf <= x_wdata_full; x_wstb <= x_wstb_full;
          x_wsfull[x_wlane] <= &x_wstrb;
        end
      end
    end
  end

  // R: lanes of the returned 64-byte beats
  logic [7:0] x_rcnt;
  logic [1:0] x_rlane_r, x_rlane;
  logic       x_rfirst;
  assign x_rlane  = x_rfirst ? xrq_rd[1:0] : x_rlane_r;
  assign x_rvalid = xr_rv && xrq_rv;
  assign x_rdata  = xr_rd[x_rlane * 128 +: 128];
  assign x_rid    = xrq_rd[XRQ-1 -: XIDW];
  assign x_rlast  = x_rcnt == xrq_rd[9:2];
  assign x_rresp  = 2'b00;
  assign xr_rr    = x_rvalid && x_rready && (x_rlane == 2'd3 || x_rlast);
  assign xrq_rr   = x_rvalid && x_rready && x_rlast;
  always_ff @(posedge xclk) begin
    if (x_crst) begin x_rcnt <= '0; x_rlane_r <= '0; x_rfirst <= 1'b1; end
    else if (x_rvalid && x_rready) begin
      x_rcnt    <= x_rlast ? '0 : x_rcnt + 1'b1;
      x_rlane_r <= x_rlane + 1'b1;
      x_rfirst  <= x_rlast;
    end
  end

  // B: a burst's response once the controller has taken all its beats
  logic [CW-1:0] x_wdone, x_wacc_c;
  always_ff @(posedge xclk) begin
    if (x_crst) begin x_wacc_s1 <= '0; x_wacc_s2 <= '0; x_wacc_c <= '0; end
    else begin x_wacc_s1 <= x_wacc_g; x_wacc_s2 <= x_wacc_s1; x_wacc_c <= g2b(x_wacc_s2); end
  end
  assign x_bvalid = xbq_rv && ((x_wacc_c - x_wdone) > CW'(xbq_rd[6:0]));
  assign x_bid    = xbq_rd[XIDW+6 -: XIDW];
  assign x_bresp  = 2'b00;
  assign xbq_rr   = x_bvalid && x_bready;
  always_ff @(posedge xclk) begin
    if (x_crst) x_wdone <= '0;
    else if (x_bvalid && x_bready) x_wdone <= x_wdone + CW'(xbq_rd[6:0]) + 1'b1;
  end

  // ================================================================ controller side (uclk)
  // Read credits (registered, one beat of margin: at most one read issues per cycle): a master
  // may issue a read while its reads not yet in its read-data FIFO (a_pend: issued, not yet
  // passed by the FIFO's write pointer) plus the FIFO's entries leave room for one more. Each read
  // takes the next slot of its master's FIFO (a_seq), which the credits keep free, and its data is
  // written there when its port returns it.
  logic a_rok, x_rok;
  logic [AOW-1:0] a_pend;
  logic [XOW-1:0] x_pend;
  logic [ASW-1:0] a_seq;
  logic [XSW-1:0] x_seq;
  always_ff @(posedge uclk) begin
    a_rok <= (a_pend + ar_used) <= AOW'(ARD - 2);
    x_rok <= (x_pend + xr_used) <= XOW'(XRD - 2);
  end

  // the arbiter: the chosen master's head command (a write with its data) goes when its port's
  // output command queue has room and, for a write, the port's output data FIFO too. Each master's
  // room (its head's port) is found alongside the pick, so go and the counts of the head that goes
  // (a_rgo: an accelerator read issued) follow the pick by a mux, not by the heads' mux (sel_x).
  logic          a_ok, x_ok, pick_x, cur_x, sel_x, go, we, part, rmw, tp;
  logic          a_room, x_room, a_rgo, x_rgo;
  logic [5:0]    run;
  logic [24:0]   addr;
  logic [DW-1:0] wd;
  logic [1:0]    oq_wv, oq_wr, of_wv, of_wr, tg_wr, tg_rv;
  logic [511:0]  of_wd;
  logic [1:0][TW-1:0] tg_rd;
  logic          rm_ret, rm_ok;
  assign a_ok = aq_rv && (aq_rd[QW-1] ? ad_rv : a_rok);
  assign x_ok = xq_rv && (xq_rd[QW-1] ? xd_rv : x_rok);
  always_comb begin
    if (!cur_x) pick_x = x_ok && (!a_ok || run >= 6'(RUN));
    else        pick_x = !a_ok || (x_ok && run < 6'(RUN));
  end
  assign sel_x = rm_busy ? rm_x : pick_x;  // a read-modify-write: its master's heads
  assign we    = sel_x ? xq_rd[QW-1] : aq_rd[QW-1];
  assign addr  = sel_x ? xq_rd[24:0] : aq_rd[24:0];
  assign wd    = sel_x ? xd_rd : ad_rd;
  assign tp    = addr[7];                  // the port: the beat's bank parity
  assign part  = we && wd[DW-1];
  assign rmw   = part;
  assign a_room = oq_wr[aq_rd[7]] && (!aq_rd[QW-1] || of_wr[aq_rd[7]]);
  assign x_room = oq_wr[xq_rd[7]] && (!xq_rd[QW-1] || of_wr[xq_rd[7]]);
  assign go    = !rm_busy && (pick_x ? x_ok && x_room : a_ok && a_room);
  assign a_rgo = !rm_busy && !pick_x && a_ok && a_room && !aq_rd[QW-1];
  assign x_rgo = !rm_busy && pick_x && x_ok && x_room && !xq_rd[QW-1];

  // read data: at most one port returns a beat in a cycle (they share the channel's data bus);
  // its tag, at the head of that port's tag FIFO, says where it goes. The core's ports carry one
  // read bus (tools/litedram/ecc_ports.py: one decoder, and each port's register behind it loads
  // every cycle, their rdata ready being tied high), so the beat is on port 0's data whichever
  // port returns it: no port mux on the valids.
  logic          rv, rp;
  logic [511:0]  rdat;
  logic [TW-1:0] rtag;
  assign rv   = |c_rdata_valid;
  assign rp   = c_rdata_valid[1];
  assign rdat = c_rdata_data[0];
  assign rtag = tg_rd[rp];

  // read-modify-write: the read goes with an RMW tag, the write's command and data stay at the
  // heads; its data merges the returning beat into the lanes not written (the same beat: the same
  // port)
  assign rm_ret = rv && rtag[TW-1];
  assign rm_ok  = rm_ret && (rm_x ? !x_hold : !a_hold);
  always_ff @(posedge uclk) begin
    if (urq) rm_busy <= 1'b0;
    else if (go && rmw) rm_busy <= 1'b1;
    else if (rm_ret) rm_busy <= 1'b0;
    if (go && rmw) rm_x <= pick_x;
  end

  assign aq_rr = (go && !pick_x && !rmw) || (rm_ret && !rm_x);
  assign xq_rr = (go && pick_x && !rmw) || (rm_ret && rm_x);
  assign ad_rr = (go && !pick_x && we && !rmw) || (rm_ret && !rm_x);
  assign xd_rr = (go && pick_x && we && !rmw) || (rm_ret && rm_x);

  // output write data: whole beats, a merged one with the read's data in the lanes not written.
  // While an RMW is in progress only its merged write enters the queues (go waits), so the merge
  // follows rm_busy, a flip-flop, rather than the returning beat's tag.
  always_comb
    for (int k = 0; k < 64; k++)
      of_wd[8 * k +: 8] = (rm_busy && !wd[512 + k]) ? rdat[8 * k +: 8] : wd[8 * k +: 8];

  // per port: the output command queue (a FIFO behind a register: the crossbar's inputs come from
  // flip-flops; a command bypasses the empty FIFO), the output write-data FIFO (pushed with the
  // write's command) and the read tags ({read-modify-write, xdma, slot}, in the port's order)
  logic [26:0]   ocn;
  logic [1:0]    opop, opw_a, opw_x, opc_a, opc_x;
  assign ocn = {rm_busy || (we && !rmw), sel_x, addr};
  for (genvar p = 0; p < 2; p++) begin : g_port
    logic        oq_rv, oq_rr, byp, oc_v;
    logic [26:0] oq_rd, oc;
    assign oq_wv[p] = (go || rm_ok) && tp == 1'(p);
    assign byp      = oq_wv[p] && !oq_rv && (!oc_v || c_cmd_ready[p]);
    assign oq_rr    = oq_rv && (!oc_v || c_cmd_ready[p]);
    otpu_sfifo #(.W(27), .DEPTH(OQ)) u_oq (.clk(uclk), .rst(urq), .wvalid(oq_wv[p] && !byp),
      .wready(oq_wr[p]), .wdata(ocn), .rvalid(oq_rv), .rready(oq_rr), .rdata(oq_rd));
    always_ff @(posedge uclk) begin
      if (urq) oc_v <= 1'b0;
      else if (!oc_v || c_cmd_ready[p]) oc_v <= oq_rv || byp;
      if (!oc_v || c_cmd_ready[p]) oc <= oq_rv ? oq_rd : ocn;
    end
    assign c_cmd_valid[p] = oc_v;
    assign c_cmd_we[p]    = oc[26];
    assign c_cmd_addr[p]  = oc[24:0];
    assign opop[p]  = oc_v && c_cmd_ready[p];
    assign opw_a[p] = opop[p] && oc[26] && !oc[25];      // a write taken, per master (n_wdone, B)
    assign opw_x[p] = opop[p] && oc[26] && oc[25];
    assign opc_a[p] = opop[p] && !oc[25];                // a command taken, per master
    assign opc_x[p] = opop[p] && oc[25];

    assign of_wv[p] = ((go && we && !rmw) || rm_ok) && tp == 1'(p);
    otpu_sfifo #(.W(512), .DEPTH(OD)) u_of (.clk(uclk), .rst(urq), .wvalid(of_wv[p]),
      .wready(of_wr[p]), .wdata(of_wd), .rvalid(c_wdata_valid[p]), .rready(c_wdata_ready[p]),
      .rdata(c_wdata_data[p]));
    assign c_wdata_we[p] = '1;

    otpu_sfifo #(.W(TW), .DEPTH(128)) u_tag (.clk(uclk), .rst(urq),
      .wvalid(go && (!we || rmw) && tp == 1'(p)), .wready(tg_wr[p]),
      .wdata({rmw, pick_x, pick_x ? ASW'(x_seq) : a_seq}), .rvalid(tg_rv[p]),
      .rready(c_rdata_valid[p]), .rdata(tg_rd[p]));
  end
  // each master's commands in the output queues (a hold waits for its own to leave)
  logic [5:0] a_nq, x_nq;
  assign a_oc = a_nq != 0;
  assign x_oc = x_nq != 0;

  // a read's data to its master's slot (dropped under its hold) or to the merge
  assign ar_wv   = rv && rtag[TW-1 -: 2] == 2'b00 && !a_hold;
  assign xr_wv   = rv && rtag[TW-1 -: 2] == 2'b01 && !x_hold;
  assign ar_slot = rtag[ASW-1:0];
  assign xr_slot = rtag[XSW-1:0];
  assign ar_wd   = rdat;
  assign xr_wd   = rdat;

  always_ff @(posedge uclk) begin
    if (urq) begin
      cur_x <= 1'b0; run <= '0; a_out <= '0; x_out <= '0; a_nq <= '0; x_nq <= '0;
    end else begin
      if (go) begin
        run   <= (pick_x != cur_x) ? 6'd1 : (run == 6'd63) ? run : run + 1'b1;
        cur_x <= pick_x;
      end
      // reads in flight (issued, data not yet back), for the holds
      a_out <= a_out + AOW'(a_rgo) - AOW'(rv && rtag[TW-1 -: 2] == 2'b00);
      x_out <= x_out + XOW'(x_rgo) - XOW'(rv && rtag[TW-1 -: 2] == 2'b01);
      a_nq  <= a_nq + 6'((go || rm_ok) && !sel_x) - 6'(opc_a[0]) - 6'(opc_a[1]);
      x_nq  <= x_nq + 6'((go || rm_ok) && sel_x) - 6'(opc_x[0]) - 6'(opc_x[1]);
    end
    // reads not yet in their FIFO, and the next slot (the FIFO restarts at slot 0 on its hold)
    if (a_hold) begin a_pend <= '0; a_seq <= '0; end
    else begin
      a_pend <= a_pend + AOW'(a_rgo) - AOW'(ar_cmt);
      if (a_rgo) a_seq <= a_seq + 1'b1;
    end
    if (x_hold) begin x_pend <= '0; x_seq <= '0; end
    else begin
      x_pend <= x_pend + XOW'(x_rgo) - XOW'(xr_cmt);
      if (x_rgo) x_seq <= x_seq + 1'b1;
    end
    if (a_hold) a_wacc <= '0; else a_wacc <= a_wacc + CW'(opw_a[0]) + CW'(opw_a[1]);
    if (x_hold) x_wacc <= '0; else x_wacc <= x_wacc + CW'(opw_x[0]) + CW'(opw_x[1]);
    a_wacc_g <= b2g(a_wacc);
    x_wacc_g <= b2g(x_wacc);
  end

  // the controller's side of the contract, checked in the build too (STATUS AXI_ERR): read data
  // from one port at a time, each beat on a port with a read outstanding. Sticky until the
  // controller's reset; into the core clock through two flip-flops.
  logic c_err = 1'b0;
  (* ASYNC_REG = "TRUE" *) logic e_s1 = 1'b0, e_s2 = 1'b0;
  always_ff @(posedge uclk)
    if (urq) c_err <= 1'b0;
    else if (&c_rdata_valid || |(c_rdata_valid & ~tg_rv)) c_err <= 1'b1;
  always_ff @(posedge clk) begin
    e_s1 <= c_err; e_s2 <= e_s1;
  end
  assign n_err = e_s2;

`ifndef SYNTHESIS
  // read data comes from one port at a time and always finds a tag and a free slot (the credits);
  // the merged write finds room (the arbiter has stopped since the read, whose own command has
  // left the queue)
  // the cycle the bridge's controller side is still in reset and the controller is not (urq
  // after urst): no command enters the queues, and the controller returns no read data
  always_ff @(posedge uclk)
    if (urq && !urst && (go || rm_ok || |oq_wv || |of_wv || |c_rdata_valid))
      $fatal(1, "otpu_mem_ch: a queue write or read data in the bridge's last reset cycle");
  always_ff @(posedge uclk) if (!urq) begin
    if (&c_rdata_valid) $error("otpu_mem_ch: read data from both ports in one cycle");
    if (go != (!rm_busy && (pick_x ? x_ok : a_ok) && oq_wr[tp] && (!we || of_wr[tp])))
      $error("otpu_mem_ch: go is not the chosen head's");
    if (a_rgo != (go && !we && !pick_x) || x_rgo != (go && !we && pick_x))
      $error("otpu_mem_ch: a read's issue is not go's");
    if (c_rdata_valid[1] && c_rdata_data[1] != c_rdata_data[0])
      $error("otpu_mem_ch: port 1's read data is not on port 0's bus");
    for (int p = 0; p < 2; p++)
      if (c_rdata_valid[p] && !tg_rv[p]) $error("otpu_mem_ch: port %0d read data without a tag", p);
    if (rm_ret && !rm_busy) $error("otpu_mem_ch: read-modify-write data without its read");
    if (rm_ok && (!oq_wr[tp] || !of_wr[tp])) $error("otpu_mem_ch: no room for the merged write");
    if (go && (!we || rmw) && !tg_wr[tp]) $error("otpu_mem_ch: tag FIFO full");
    if (a_pend + ar_used > AOW'(ARD)) $error("otpu_mem_ch: accelerator read slots overrun");
    if (x_pend + xr_used > XOW'(XRD)) $error("otpu_mem_ch: XDMA read slots overrun");
  end
  // a hold ends with none of its master's commands left to count (n_wdone / B restart from 0)
  logic a_hold_q, x_hold_q;
  always_ff @(posedge uclk) begin
    a_hold_q <= a_hold;
    x_hold_q <= x_hold;
    if (a_hold_q && !a_hold && a_oc) $error("otpu_mem_ch: accelerator hold over, command pending");
    if (x_hold_q && !x_hold && x_oc) $error("otpu_mem_ch: XDMA hold over, command pending");
  end
`endif
endmodule

// Synchronous FIFO, first-word fall-through, distributed RAM (the channel bridge's queues).
// RO = 1: the head entry from a register, so no path from rdata starts at the read pointer and
// goes through the RAM; the same cycles (rdata differs only while rvalid is low).
module otpu_sfifo #(
  parameter int W = 8,
  parameter int DEPTH = 16,
  parameter bit RO = 1'b0
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         wvalid,
  output logic         wready,
  input  logic [W-1:0] wdata,
  output logic         rvalid,
  input  logic         rready,
  output logic [W-1:0] rdata
);
  localparam int AW = $clog2(DEPTH);
  (* ram_style = "distributed" *) logic [W-1:0] mem [DEPTH];
  logic [AW:0] wp, rp;
  assign wready = (wp - rp) != (AW + 1)'(DEPTH);
  assign rvalid = wp != rp;
  always_ff @(posedge clk) begin
    if (rst) begin wp <= '0; rp <= '0; end
    else begin
      if (wvalid && wready) wp <= wp + 1'b1;
      if (rvalid && rready) rp <= rp + 1'b1;
    end
  end
  always_ff @(posedge clk) if (wvalid && wready) mem[wp[AW-1:0]] <= wdata;
  if (RO) begin : g_ro
    // the next head: on a pop the entry after it (this cycle's write if there is none), into an
    // empty FIFO this cycle's write
    logic [W-1:0]  head;
    logic [AW-1:0] rp1;
    logic          one;
    assign rp1 = rp[AW-1:0] + 1'b1;
    assign one = (wp - rp) == (AW + 1)'(1);
    always_ff @(posedge clk) begin
      if (rvalid && rready) head <= one ? wdata : mem[rp1];
      else if (!rvalid)     head <= wdata;
    end
    assign rdata = head;
  end else begin : g_ra
    assign rdata = mem[rp[AW-1:0]];
  end
endmodule
