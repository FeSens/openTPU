// LD / ST: move 32-bit words between the slice DRAM and TMEM using the DRAM burst port (B).
// The transfer is cut into W = min(D/4, LANES)-word segments aligned in DRAM; each cycle moves
// one segment, requested as the D-byte chunk that holds it (B requests are chunk aligned), so
// the lanes need no shifter: lane l carries word l of the segment, and the partial segments at
// the ends are masked. The segment's words sit in consecutive TMEM words, hence distinct banks.
// The DRAM port may refuse a request (b_gnt low) and read data may take any time to return (in
// order). An ST completes once the DRAM has acknowledged all its writes (wr_idle).
module otpu_dma
  import otpu_pkg::*;
#(
  parameter int D     = 32,
  parameter int LANES = 8
) (
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    start,
  input  cmd_t                    cmd,
  output logic                    rdy,
  output logic                    done,
  // DRAM port B (the DMA has priority on it; read responses are routed back by tag)
  output logic                    b_req,
  input  logic                    b_gnt,      // the request is taken this cycle
  output logic                    b_we,
  output logic [D/4-1:0]          b_wmask,
  output logic [D*8-1:0]          b_wdata,
  output logic [31:0]             b_addr,
  input  logic                    b_rvalid,
  input  logic [D*8-1:0]          b_rdata,
  input  logic                    wr_idle,    // no DRAM write outstanding
  // TMEM lanes (read port A, write port)
  output logic [LANES-1:0]        t_ren,
  output logic [LANES-1:0][31:0]  t_raddr,
  input  logic [LANES-1:0][31:0]  t_rdata,
  output logic [LANES-1:0]        t_wen,
  output logic [LANES-1:0][31:0]  t_waddr,
  output logic [LANES-1:0][31:0]  t_wdata
);
  localparam int CW = D / 4;                          // words per chunk
  localparam int W  = (CW < LANES) ? CW : LANES;      // words per segment
  localparam int SWL = $clog2(W);
  logic        busy, is_st, st_pend, ackw;
  assign rdy = !busy;
  logic [31:0] dw, tm, n;              // DRAM word address, TMEM address, words
  logic [31:0] nseg, iss, cmp;         // segments: total, issued, completed (LD)
  logic [31:0] iw, rw, pw;             // DRAM word address of the segment issued / received / pending
  wire         adv = !st_pend || b_gnt;  // ST: the read -> write pipeline moves
  logic [31:0]  rofs;                   // LD: TMEM address of the segment at rw (tm + rw - dw)
  logic [W-1:0] rmask;                  // LD: lanes of the segment at rw inside the range
                                        // (registered: the TMEM arbiter sees these lanes)

  function automatic logic in_rng(input logic [31:0] a);
    return (a >= dw) && (a < dw + n);
  endfunction
  function automatic logic [31:0] chunk_of(input logic [31:0] a);
    return a & ~32'(CW - 1);
  endfunction
  function automatic int pos_of(input logic [31:0] a);     // segment index within its chunk
    return int'((a % CW) / W);
  endfunction

  always_comb begin
    b_req = 1'b0; b_we = 1'b0; b_wmask = '0; b_wdata = '0; b_addr = '0;
    t_ren = '0; t_raddr = '0; t_wen = '0; t_waddr = '0; t_wdata = '0;
    if (busy && !is_st && !ackw) begin
      if (iss < nseg) begin
        b_req = 1'b1;
        b_addr = chunk_of(iw);
      end
      if (b_rvalid) begin
        for (int l = 0; l < W; l++) begin
          if (rmask[l]) begin
            t_wen[l] = 1'b1;
            t_waddr[l] = rofs + 32'(l);
            t_wdata[l] = b_rdata[32 * (pos_of(rw) * W + l) +: 32];
          end
        end
      end
    end
    if (busy && is_st && !ackw) begin
      if (iss < nseg && adv) begin
        for (int l = 0; l < W; l++) begin
          if (in_rng(iw + 32'(l))) begin
            t_ren[l] = 1'b1;
            t_raddr[l] = tm + iw + 32'(l) - dw;
          end
        end
      end
      if (st_pend) begin
        b_req = 1'b1;
        b_we = 1'b1;
        b_addr = chunk_of(pw);
        for (int l = 0; l < W; l++) begin
          if (in_rng(pw + 32'(l))) begin
            b_wmask[pos_of(pw) * W + l] = 1'b1;
            b_wdata[32 * (pos_of(pw) * W + l) +: 32] = t_rdata[l];
          end
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    done <= 1'b0;
    if (rst) begin
      busy <= 1'b0;
      st_pend <= 1'b0;
      ackw <= 1'b0;
    end else if (start) begin
      logic [31:0] a, s0, s1;
      a = cmd.w1 >> 2;
      s0 = a & ~32'(W - 1);
      s1 = (a + cmd.w3 - 1) & ~32'(W - 1);
      is_st <= (cmd.op == OP_ST);
      dw <= a;
      tm <= cmd.w2;
      n <= cmd.w3;
      nseg <= ((s1 - s0) >> SWL) + 1;
      iss <= '0; cmp <= '0;
      iw <= s0; rw <= s0;
      for (int l = 0; l < W; l++) rmask[l] <= (s0 + 32'(l) >= a) && (s0 + 32'(l) < a + cmd.w3);
      rofs <= cmd.w2 + s0 - a;
      st_pend <= 1'b0;
      ackw <= 1'b0;
      if (cmd.w3 == 0) done <= 1'b1;
      else busy <= 1'b1;
    end else if (ackw) begin
      if (wr_idle) begin
        ackw <= 1'b0;
        busy <= 1'b0;
        done <= 1'b1;
      end
    end else if (busy) begin
      if (!is_st) begin
        if (iss < nseg && b_gnt) begin
          iss <= iss + 1;
          iw <= iw + W;
        end
        if (b_rvalid) begin
          cmp <= cmp + 1;
          rw <= rw + W;
          rofs <= rofs + W;
          for (int l = 0; l < W; l++) rmask[l] <= in_rng(rw + W + 32'(l));
          if (cmp + 1 == nseg) begin
            busy <= 1'b0;
            done <= 1'b1;
          end
        end
      end else if (adv) begin
        st_pend <= (iss < nseg);
        pw <= iw;
        if (iss < nseg) begin
          iss <= iss + 1;
          iw <= iw + W;
        end
        if (st_pend && iss == nseg) begin       // the last write is taken this cycle
          st_pend <= 1'b0;
          ackw <= 1'b1;
        end
      end
    end
  end
endmodule
