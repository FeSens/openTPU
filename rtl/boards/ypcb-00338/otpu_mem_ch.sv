// One DDR3 channel's user side: the accelerator's native master (otpu_native_dram, core clock,
// one 64-byte beat per command) and XDMA's AXI master (host loads and reads, axi_aclk, 128-bit)
// onto one native port of the channel's controller (uclk: LiteDRAM's sys). With otpu_axi_split2
// it replaced the MIG builds' SmartConnect and the MIG's AXI front end.
//
// Each master's commands cross into uclk through its own asynchronous FIFOs (commands, write
// data) and its read data comes back through another; an arbiter in uclk takes up to RUN commands
// from one master before it lets the other in, one command per cycle, and each master's commands
// in its order. The accelerator's write data comes in write-command order, before or after its
// command: a write goes when both are at the heads of their FIFOs. The controller returns read
// data in command order and cannot hold it back, so one tag per read (its master) routes the data
// back, and a master's read is issued only when its read-data FIFO has room for it and for every
// read still in flight (credits). The accelerator's read data leaves without backpressure too
// (its master reserves room for its reads), from a register.
//
// The controller port has LiteDRAM's native-port semantics: commands (we, beat), write data taken
// (wdata ready) some cycles after its command, in write-command order, and read data in command
// order. LiteDRAM's crossbar takes the write data when the bank asks for it without looking at
// wdata valid, so a write's data enters the output data FIFO (only the controller's reset clears
// it) in the cycle its command enters the output command register (2 entries, so the arbiter
// sees only registered state): the data is there before the controller can take the command, and
// the per-master holds cannot take either back. (LiteDRAMNativePortECC puts a one-beat register
// in front of the crossbar that takes the head as soon as it is empty; it is loaded well before
// the bank asks, which is at least 4 cycles after the command. Its rdata ready must be tied high.)
//
// Partial beats (any byte not written): LiteDRAM's ECC port rejects partial writes, and the board
// has no DM pins, so the read-modify-write is done here. The arbiter issues a read of the beat and
// stops; the write's command and data stay at the heads of their master's FIFOs; when the read's
// data comes back (its tag), the old bytes fill the lanes not written and the whole beat is
// written, and only then does the arbiter go on. So nothing reaches the controller between the
// read and the write: the read comes after every command issued before it, the write before every
// command issued after it, and the controller keeps one port's commands in order (LiteDRAM locks a
// port to one bank until that bank's queue has drained; each bank machine is a FIFO). A partial
// beat costs about one read latency of the whole channel.
//
// n_wdone counts the accelerator's write beats the controller has taken (the command handshake;
// the data is then already in the output FIFO), gray-coded across into the core clock. Every
// command issued after that, from either master, reaches the controller behind the write, so the
// write is visible to both masters. An XDMA write burst gets its B the same way, once all of its
// 64-byte beats are taken. XDMA's 16-byte beats are packed into 64-byte beats per burst (a partial
// beat at either end is a partial write).
//
// Addresses: n_caddr and bits 30:6 of XDMA's byte address select the 64-byte beat in the channel
// (2 GiB; XDMA's bit 31, the channel, is dropped by otpu_axi_split2). XDMA's bursts are INCR of
// 16-byte beats (AxSIZE 4) and may start anywhere in a 64-byte beat; narrow bursts are not
// supported (XDMA issues none). Its reads return in order (legal for any IDs); responses are OKAY.
//
// Resets: each master's path has a hold in uclk -- the controller's reset, or the master's own
// reset (synchronized, and kept up until the hold is seen in the master's clock, so that a reset
// of a cycle gets a whole hold too), then 16 cycles, and on until its reads in flight have come
// back (their data is dropped), a read-modify-write of its own is done and no command of its own
// is left in the output command register (a write counted after the hold would count one the
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
  parameter int OD   = 32         // output write-data FIFO: writes issued, data not yet taken
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
  // controller native port, uclk
  input  logic            uclk,
  input  logic            urst,
  output logic            c_cmd_valid,
  input  logic            c_cmd_ready,
  output logic            c_cmd_we,
  output logic [24:0]     c_cmd_addr,
  output logic            c_wdata_valid,
  input  logic            c_wdata_ready,
  output logic [511:0]    c_wdata_data,
  output logic [63:0]     c_wdata_we,     // 1 = write the byte (all ones: whole beats)
  input  logic            c_rdata_valid,  // in command order, no backpressure
  input  logic [511:0]    c_rdata_data
);
  localparam int QW = 26;                  // command: {write, beat[24:0]}
  localparam int DW = 577;                 // write data: {partial, byte enables[63:0], data[511:0]}
  localparam int CW = 16;                  // write-accept counters (beats, modulo 2^16)
  localparam int AOW = $clog2(ARD) + 1;
  localparam int XOW = $clog2(XRD) + 1;
  initial if (ARD + XRD + 1 > 128) $fatal(1, "otpu_mem_ch: ARD + XRD reads in flight exceed the tag FIFO");

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
  logic [1:0]  on;                         // output command register: entries
  logic [26:0] oc0, oc1;                   //   {write, xdma, beat}
  logic a_oc, x_oc;                        // a command of the master's in it
  assign a_oc = (on[0] && !oc0[25]) || (on[1] && !oc1[25]);
  assign x_oc = (on[0] && oc0[25]) || (on[1] && oc1[25]);
  always_ff @(posedge uclk) begin
    a_rs1 <= a_req; a_rs2 <= a_rs1;
    x_rs1 <= x_req; x_rs2 <= x_rs1;
    a_hcnt <= (urst || a_rs2) ? 4'hF : (a_hcnt != 0) ? a_hcnt - 1'b1 : a_hcnt;
    x_hcnt <= (urst || x_rs2) ? 4'hF : (x_hcnt != 0) ? x_hcnt - 1'b1 : x_hcnt;
    a_hold <= urst || a_rs2 ||
              (a_hold && (a_hcnt != 0 || a_out != 0 || (rm_busy && !rm_x) || a_oc));
    x_hold <= urst || x_rs2 ||
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

  otpu_afifo #(.W(QW), .DEPTH(32)) u_aq (.wclk(clk), .wrst(a_hs2), .wvalid(aq_wv), .wready(aq_wr),
    .wdata(aq_wd), .wused(unused_aq), .rclk(uclk), .rrst(a_hold), .rvalid(aq_rv), .rready(aq_rr),
    .rdata(aq_rd));
  otpu_afifo #(.W(DW), .DEPTH(32)) u_ad (.wclk(clk), .wrst(a_hs2), .wvalid(ad_wv), .wready(ad_wr),
    .wdata(ad_wd), .wused(unused_ad), .rclk(uclk), .rrst(a_hold), .rvalid(ad_rv), .rready(ad_rr),
    .rdata(ad_rd));
  otpu_afifo #(.W(512), .DEPTH(ARD)) u_ar (.wclk(uclk), .wrst(a_hold), .wvalid(ar_wv), .wready(ar_wr),
    .wdata(ar_wd), .wused(ar_used), .rclk(clk), .rrst(a_hs2), .rvalid(ar_rv), .rready(ar_rr),
    .rdata(ar_rd));
  otpu_afifo #(.W(QW), .DEPTH(16)) u_xq (.wclk(xclk), .wrst(x_hs2), .wvalid(xq_wv), .wready(xq_wr),
    .wdata(xq_wd), .wused(unused_xq), .rclk(uclk), .rrst(x_hold), .rvalid(xq_rv), .rready(xq_rr),
    .rdata(xq_rd));
  otpu_afifo #(.W(DW), .DEPTH(16)) u_xd (.wclk(xclk), .wrst(x_hs2), .wvalid(xd_wv), .wready(xd_wr),
    .wdata(xd_wd), .wused(unused_xd), .rclk(uclk), .rrst(x_hold), .rvalid(xd_rv), .rready(xd_rr),
    .rdata(xd_rd));
  otpu_afifo #(.W(512), .DEPTH(XRD)) u_xr (.wclk(uclk), .wrst(x_hold), .wvalid(xr_wv), .wready(xr_wr),
    .wdata(xr_wd), .wused(xr_used), .rclk(xclk), .rrst(x_hs2), .rvalid(xr_rv), .rready(xr_rr),
    .rdata(xr_rd));

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
  otpu_sfifo #(.W(32), .DEPTH(4)) u_xai (.clk(xclk), .rst(x_crst), .wvalid(xai_wv), .wready(xai_wr),
    .wdata({x_arn, x_araddr[30:6]}), .rvalid(xai_rv), .rready(xai_rr), .rdata(xai_rd));
  otpu_sfifo #(.W(27), .DEPTH(4)) u_xwi (.clk(xclk), .rst(x_crst), .wvalid(xwi_wv), .wready(xwi_wr),
    .wdata({x_awaddr[5:4], x_awaddr[30:6]}), .rvalid(xwi_rv), .rready(xwi_rr), .rdata(xwi_rd));
  otpu_sfifo #(.W(XRQ), .DEPTH(16)) u_xrq (.clk(xclk), .rst(x_crst), .wvalid(xrq_wv), .wready(xrq_wr),
    .wdata({x_arid, x_arlen, x_araddr[5:4]}), .rvalid(xrq_rv), .rready(xrq_rr), .rdata(xrq_rd));
  otpu_sfifo #(.W(XIDW + 7), .DEPTH(16)) u_xbq (.clk(xclk), .rst(x_crst), .wvalid(xbq_wv), .wready(xbq_wr),
    .wdata({x_awid, x_awn}), .rvalid(xbq_rv), .rready(xbq_rr), .rdata(xbq_rd));
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
  assign xd_wd    = {~&x_wstb_full, x_wstb_full, x_wdata_full};
  assign xai_rr   = xq_wv && !x_selw && x_rlastb;
  assign xwi_rr   = x_wvalid && x_wready && x_wlast;
  always_ff @(posedge xclk) begin
    if (x_crst) begin
      x_roff <= '0; x_woff <= '0; x_wlane_r <= '0; x_wfirst <= 1'b1; x_wbuf <= '0; x_wstb <= '0;
    end else begin
      if (xq_wv && !x_selw) x_roff <= x_rlastb ? '0 : x_roff + 1'b1;
      if (x_wvalid && x_wready) begin
        x_wfirst  <= x_wlast;
        x_wlane_r <= x_wlane + 1'b1;
        if (x_wpush) begin
          x_woff <= x_wlast ? '0 : x_woff + 1'b1;
          x_wbuf <= '0; x_wstb <= '0;
        end else begin
          x_wbuf <= x_wdata_full; x_wstb <= x_wstb_full;
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
  // may issue a read while its reads in flight plus its FIFO's entries leave room for one more.
  logic a_rok, x_rok;
  always_ff @(posedge uclk) begin
    a_rok <= (a_out + ar_used) <= AOW'(ARD - 2);
    x_rok <= (x_out + xr_used) <= XOW'(XRD - 2);
  end

  // the arbiter: the chosen master's head command (a write with its data) goes when the output
  // command register has room and, for a write, the output data FIFO too
  logic          a_ok, x_ok, pick_x, cur_x, sel_x, go, we, part, rmw;
  logic [5:0]    run;
  logic [24:0]   addr;
  logic [DW-1:0] wd;
  logic          of_wv, of_wr, of_rv;      // output write-data FIFO
  logic [511:0]  of_wd, of_rd;
  logic          rm_ret, rm_ok;
  logic          tg_wr, tg_rv;             // read tags
  logic [1:0]    tg_rd;
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
  assign part  = we && wd[DW-1];
  assign rmw   = part;
  assign go    = !rm_busy && (pick_x ? x_ok : a_ok) && !on[1] && (!we || of_wr);

  // read-modify-write: the read goes with an RMW tag, the write's command and data stay at the
  // heads; its data merges the returning beat into the lanes not written
  assign rm_ret = c_rdata_valid && tg_rd[1];
  assign rm_ok  = rm_ret && (rm_x ? !x_hold : !a_hold);
  always_ff @(posedge uclk) begin
    if (urst) rm_busy <= 1'b0;
    else if (go && rmw) rm_busy <= 1'b1;
    else if (rm_ret) rm_busy <= 1'b0;
    if (go && rmw) rm_x <= pick_x;
  end

  assign aq_rr = (go && !pick_x && !rmw) || (rm_ret && !rm_x);
  assign xq_rr = (go && pick_x && !rmw) || (rm_ret && rm_x);
  assign ad_rr = (go && !pick_x && we && !rmw) || (rm_ret && !rm_x);
  assign xd_rr = (go && pick_x && we && !rmw) || (rm_ret && rm_x);

  // output write data: pushed with the write's command (whole beats, a merged one with the read's
  // data in the lanes not written)
  assign of_wv = (go && we && !rmw) || rm_ok;
  always_comb
    for (int k = 0; k < 64; k++)
      of_wd[8 * k +: 8] = (rm_ret && !wd[512 + k]) ? c_rdata_data[8 * k +: 8] : wd[8 * k +: 8];
  assign c_wdata_we = '1;
  otpu_sfifo #(.W(512), .DEPTH(OD)) u_of (.clk(uclk), .rst(urst), .wvalid(of_wv), .wready(of_wr),
    .wdata(of_wd), .rvalid(of_rv), .rready(c_wdata_ready), .rdata(of_rd));
  assign c_wdata_valid = of_rv;
  assign c_wdata_data  = of_rd[511:0];

  // output command register (2 entries: go needs no c_cmd_ready): {write, xdma, beat}
  logic [26:0] ocn;
  logic        opush, opop;
  assign opush = go || rm_ok;
  assign opop  = on[0] && c_cmd_ready;
  assign ocn   = {rm_ok || (we && !rmw), sel_x, addr};
  always_ff @(posedge uclk) begin
    if (urst) on <= 2'b00;
    else if (opush && !opop) on <= {on[0], 1'b1};
    else if (!opush && opop) on <= {1'b0, on[1]};
    if (opop) oc0 <= on[1] ? oc1 : ocn;
    else if (!on[0]) oc0 <= ocn;
    if (opush && on[0] && !opop) oc1 <= ocn;
  end
  assign c_cmd_valid   = on[0];
  assign c_cmd_we      = oc0[26];
  assign c_cmd_addr    = oc0[24:0];

  // each read's tag, in command order: {read-modify-write, xdma}; its data to that master (dropped
  // under its hold) or to the merge
  otpu_sfifo #(.W(2), .DEPTH(128)) u_tag (.clk(uclk), .rst(urst), .wvalid(go && (!we || rmw)),
    .wready(tg_wr), .wdata({rmw, pick_x}), .rvalid(tg_rv), .rready(c_rdata_valid), .rdata(tg_rd));
  assign ar_wv = c_rdata_valid && tg_rd == 2'b00 && !a_hold;
  assign xr_wv = c_rdata_valid && tg_rd == 2'b01 && !x_hold;
  assign ar_wd = c_rdata_data;
  assign xr_wd = c_rdata_data;

  always_ff @(posedge uclk) begin
    if (urst) begin
      cur_x <= 1'b0; run <= '0; a_out <= '0; x_out <= '0;
    end else begin
      if (go) begin
        run   <= (pick_x != cur_x) ? 6'd1 : (run == 6'd63) ? run : run + 1'b1;
        cur_x <= pick_x;
      end
      a_out <= a_out + AOW'(go && !we && !pick_x) - AOW'(c_rdata_valid && tg_rd == 2'b00);
      x_out <= x_out + XOW'(go && !we && pick_x) - XOW'(c_rdata_valid && tg_rd == 2'b01);
    end
    if (a_hold) a_wacc <= '0; else if (opop && oc0[26] && !oc0[25]) a_wacc <= a_wacc + 1'b1;
    if (x_hold) x_wacc <= '0; else if (opop && oc0[26] && oc0[25]) x_wacc <= x_wacc + 1'b1;
    a_wacc_g <= b2g(a_wacc);
    x_wacc_g <= b2g(x_wacc);
  end

`ifndef SYNTHESIS
  // read data always finds room (the credits) and a tag; the merged write finds room (the
  // arbiter has stopped since the read, whose own command has left the register)
  always_ff @(posedge uclk) if (!urst && c_rdata_valid) begin
    if (!tg_rv) $error("otpu_mem_ch: read data without a tag");
    if (ar_wv && !ar_wr) $error("otpu_mem_ch: accelerator read-data FIFO full");
    if (xr_wv && !xr_wr) $error("otpu_mem_ch: XDMA read-data FIFO full");
    if (rm_ret && !rm_busy) $error("otpu_mem_ch: read-modify-write data without its read");
    if (rm_ok && (on != 0 || !of_wr)) $error("otpu_mem_ch: no room for the merged write");
  end
  always_ff @(posedge uclk) if (!urst && go && (!we || rmw) && !tg_wr) $error("otpu_mem_ch: tag FIFO full");
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
module otpu_sfifo #(
  parameter int W = 8,
  parameter int DEPTH = 16
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
  assign rdata  = mem[rp[AW-1:0]];
  always_ff @(posedge clk) begin
    if (rst) begin wp <= '0; rp <= '0; end
    else begin
      if (wvalid && wready) wp <= wp + 1'b1;
      if (rvalid && rready) rp <= rp + 1'b1;
    end
  end
  always_ff @(posedge clk) if (wvalid && wready) mem[wp[AW-1:0]] <= wdata;
endmodule
