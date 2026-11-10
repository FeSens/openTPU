// Asynchronous FIFO, first-word fall-through: gray-coded pointers, each synchronized by two
// flip-flops into the other clock, and a distributed-RAM array (asynchronous read), so rdata is
// the head entry while rvalid is high. DEPTH is a power of two.
//
// wused counts the entries the write side still has to consider occupied: what it wrote minus
// what the read side has popped as last seen through the synchronizer (so at most a few cycles
// stale, always on the safe side). The channel bridge uses it for read credits. rused counts the
// entries the read side sees: the write pointer as synchronized minus what it has popped (as stale,
// on the same safe side). The channel bridge batches XDMA's writes on it.
//
// OOO = 1: entries are written out of order, each at its slot (wslot: the entry's place in the
// FIFO's order, modulo DEPTH), and enter the FIFO in slot order: the write pointer passes an entry
// once it and every entry before it are written (at most one a cycle; wcommit), so the read side
// sees them in order. It passes an entry the cycle after its write, from the slots' flags alone,
// so no write-side input reaches wcommit (the bridge's read credits follow it). The writer
// reserves the slots (the channel bridge's read credits): a write is not checked against wready,
// and a slot is written once per pass.
//
// RWR = 1: wready is a flip-flop, so the writer's enables start from flip-flops: the room after
// this cycle's write, against the read pointer as synchronized now (at most a cycle more
// conservative than wused), low from the cycle after wrst rises to the cycle after it falls. In
// wrst's first cycle it keeps the last cycle's value; the channel bridge's XDMA side is in its own
// reset then (x_crst), except in the first cycle of a hold the controller's reset raised, when a
// beat it takes is lost with those already in the FIFO. The RAM's write address is a copy of the
// write pointer of its own then, replicated by synthesis (MAX_FANOUT) over the RAM's width (the
// XDMA side at 250 MHz, PCIe Gen2).
//
// WCP > 1 (RWR): the RAM in WCP column groups of W / WCP bits, each written from copies of its own
// of wready's flip-flop and of the write address (DONT_TOUCH: synthesis merged the address copy
// above with the write pointer, wbin, and the merged register drove 171 RAM address pins, wready
// 297 enables, over 100 rows of SLR0: bcu1525-s2's s2syn, -0.225 ns at 300 MHz). Cycle for cycle
// the FIFO of WCP 1; WCP 1 (the default) is the RAM and its enables as above.
//
// In order (OOO = 0) the RAM writes the slot at the tail every cycle there is room (wready), and
// a write only moves the tail: the RAM's write enables come from the pointers' registers (RWR:
// from wready's flip-flop), not from the writer's valid (on the card the native adapter's write
// issue, 9 levels from its hazard registers, ran into u_ad's 782 write enables). With room the
// slot at the tail is free (wused is never under what the read side holds), and the read side
// reads a slot as valid only once the write pointer, as synchronized, has passed it. (The room
// is this cycle's wready, not one a cycle older: a write wready takes is always written.)
module otpu_afifo #(
  parameter int W = 32,
  parameter int DEPTH = 16,
  parameter bit OOO = 1'b0,
  parameter bit RWR = 1'b0,
  parameter int WCP = 1                     // RWR: the RAM's write enable and address in copies
) (
  input  logic         wclk,
  input  logic         wrst,
  input  logic         wvalid,
  output logic         wready,
  input  logic [W-1:0] wdata,
  input  logic [$clog2(DEPTH)-1:0] wslot,    // OOO: the entry's slot
  output logic         wcommit,              // OOO: an entry entered the FIFO
  output logic [$clog2(DEPTH):0] wused,
  input  logic         rclk,
  input  logic         rrst,
  output logic         rvalid,
  input  logic         rready,
  output logic [W-1:0] rdata,
  output logic [$clog2(DEPTH):0] rused
);
  localparam int AW = $clog2(DEPTH);
  initial if (DEPTH != (1 << AW) || DEPTH < 4) $fatal(1, "otpu_afifo: DEPTH must be a power of two >= 4");
  initial if (WCP < 1 || (WCP > 1 && (!RWR || OOO || W % WCP != 0)))
    $fatal(1, "otpu_afifo: WCP %0d (1, or with RWR and in order, a divisor of W)", WCP);

  (* ram_style = "distributed" *) logic [W-1:0] mem [DEPTH];
  logic [AW:0] wbin, wgray, rbin, rgray;
  (* ASYNC_REG = "TRUE" *) logic [AW:0] rgray_w1, rgray_w2;   // read pointer, in wclk
  (* ASYNC_REG = "TRUE" *) logic [AW:0] wgray_r1, wgray_r2;   // write pointer, in rclk

  function automatic logic [AW:0] g2b(input logic [AW:0] g);
    logic [AW:0] b;
    b[AW] = g[AW];
    for (int i = AW - 1; i >= 0; i--) b[i] = b[i + 1] ^ g[i];
    return b;
  endfunction

  // write side
  logic [AW:0] rbin_w;
  logic        wput, wadv;
  logic [AW-1:0] waddr, wa;
  logic [DEPTH-1:0] wvld;                  // OOO: slots written, not yet passed
  assign rbin_w = g2b(rgray_w2);
  assign wused  = wbin - rbin_w;
  if (RWR) begin : g_rwr
    logic wrdy = 1'b0;
    always_ff @(posedge wclk)
      wrdy <= !wrst && ((wbin + (AW + 1)'(wput)) - rbin_w) != (AW + 1)'(DEPTH);
    assign wready = wrdy;
    (* max_fanout = 32 *) logic [AW-1:0] wa_q;
    always_ff @(posedge wclk)
      if (wrst) wa_q <= '0;
      else if (wadv) wa_q <= wbin[AW-1:0] + 1'b1;
    assign wa = wa_q;
  end else begin : g_cwr
    assign wready = !wrst && (wused != (AW + 1)'(DEPTH));
    assign wa = wbin[AW-1:0];
  end
  assign wput   = OOO ? (wvalid && !wrst) : (wvalid && wready);
  assign waddr  = OOO ? wslot : wa;
  assign wadv   = OOO ? (!wrst && wvld[wbin[AW-1:0]])
                      : (wvalid && wready);
  assign wcommit = OOO && wadv;
  always_ff @(posedge wclk) begin
    if (wrst) begin
      wbin <= '0; wgray <= '0; rgray_w1 <= '0; rgray_w2 <= '0;
    end else begin
      rgray_w1 <= rgray; rgray_w2 <= rgray_w1;
      if (wadv) begin
        wbin  <= wbin + 1'b1;
        wgray <= (wbin + 1'b1) ^ ((wbin + 1'b1) >> 1);
      end
    end
  end
  always_ff @(posedge wclk) if (WCP == 1 && (OOO ? wput : wready)) mem[waddr] <= wdata;
  logic [W-1:0] rdata_c;                   // WCP > 1: the column groups' heads
  if (WCP > 1) begin : g_wcp
    localparam int GW = W / WCP;
    (* dont_touch = "true" *) logic [WCP-1:0]         we_c = '0;
    (* dont_touch = "true" *) logic [WCP-1:0][AW-1:0] wa_c = '0;
    always_ff @(posedge wclk) begin
      we_c <= {WCP{!wrst && ((wbin + (AW + 1)'(wput)) - rbin_w) != (AW + 1)'(DEPTH)}};
      if (wrst) wa_c <= '0;
      else if (wadv) wa_c <= {WCP{wbin[AW-1:0] + 1'b1}};
    end
    for (genvar g = 0; g < WCP; g++) begin : g_c
      (* ram_style = "distributed" *) logic [GW-1:0] cmem [DEPTH];
      always_ff @(posedge wclk) if (we_c[g]) cmem[wa_c[g]] <= wdata[g * GW +: GW];
      assign rdata_c[g * GW +: GW] = cmem[rbin[AW-1:0]];
    end
  end else begin : g_w1
    assign rdata_c = '0;
  end
  if (OOO) begin : g_ooo
    always_ff @(posedge wclk) begin
      if (wrst) wvld <= '0;
      else begin
        if (wput) wvld[wslot] <= 1'b1;
        if (wadv) wvld[wbin[AW-1:0]] <= 1'b0;
      end
    end
`ifndef SYNTHESIS
    always_ff @(posedge wclk) if (!wrst && wput && wvld[wslot])
      $error("otpu_afifo: slot %0d written twice", wslot);
`endif
  end else begin : g_fifo
    assign wvld = '0;
  end

  // read side
  assign rvalid = !rrst && (rgray != wgray_r2);
  assign rdata  = WCP > 1 ? rdata_c : mem[rbin[AW-1:0]];
  assign rused  = g2b(wgray_r2) - rbin;
  always_ff @(posedge rclk) begin
    if (rrst) begin
      rbin <= '0; rgray <= '0; wgray_r1 <= '0; wgray_r2 <= '0;
    end else begin
      wgray_r1 <= wgray; wgray_r2 <= wgray_r1;
      if (rvalid && rready) begin
        rbin  <= rbin + 1'b1;
        rgray <= (rbin + 1'b1) ^ ((rbin + 1'b1) >> 1);
      end
    end
  end
endmodule
