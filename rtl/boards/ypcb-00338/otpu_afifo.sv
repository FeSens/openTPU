// Asynchronous FIFO, first-word fall-through: gray-coded pointers, each synchronized by two
// flip-flops into the other clock, and a distributed-RAM array (asynchronous read), so rdata is
// the head entry while rvalid is high. DEPTH is a power of two.
//
// wused counts the entries the write side still has to consider occupied: what it wrote minus
// what the read side has popped as last seen through the synchronizer (so at most a few cycles
// stale, always on the safe side). The channel bridge uses it for read credits.
module otpu_afifo #(
  parameter int W = 32,
  parameter int DEPTH = 16
) (
  input  logic         wclk,
  input  logic         wrst,
  input  logic         wvalid,
  output logic         wready,
  input  logic [W-1:0] wdata,
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
  assign rbin_w = g2b(rgray_w2);
  assign wused  = wbin - rbin_w;
  assign wready = !wrst && (wused != (AW + 1)'(DEPTH));
  always_ff @(posedge wclk) begin
    if (wrst) begin
      wbin <= '0; wgray <= '0; rgray_w1 <= '0; rgray_w2 <= '0;
    end else begin
      rgray_w1 <= rgray; rgray_w2 <= rgray_w1;
      if (wvalid && wready) begin
        wbin  <= wbin + 1'b1;
        wgray <= (wbin + 1'b1) ^ ((wbin + 1'b1) >> 1);
      end
    end
  end
  always_ff @(posedge wclk) if (wvalid && wready) mem[wbin[AW-1:0]] <= wdata;

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
