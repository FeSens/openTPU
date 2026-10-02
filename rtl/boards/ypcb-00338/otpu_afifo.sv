// Asynchronous FIFO, first-word fall-through: gray-coded pointers, each synchronized by two
// flip-flops into the other clock, and a distributed-RAM array (asynchronous read), so rdata is
// the head entry while rvalid is high. DEPTH is a power of two.
//
// wused counts the entries the write side still has to consider occupied: what it wrote minus
// what the read side has popped as last seen through the synchronizer (so at most a few cycles
// stale, always on the safe side). The channel bridge uses it for read credits.
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
module otpu_afifo #(
  parameter int W = 32,
  parameter int DEPTH = 16,
  parameter bit OOO = 1'b0,
  parameter bit RWR = 1'b0
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
  output logic [W-1:0] rdata
);
  localparam int AW = $clog2(DEPTH);
  initial if (DEPTH != (1 << AW) || DEPTH < 4) $fatal(1, "otpu_afifo: DEPTH must be a power of two >= 4");

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
  always_ff @(posedge wclk) if (wput) mem[waddr] <= wdata;
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
  assign rdata  = mem[rbin[AW-1:0]];
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
