// LD / ST: move 32-bit words between the slice DRAM and TMEM using the DRAM burst port (B).
// DRAM side: every D-byte chunk the transfer touches is requested once (B requests are chunk
// aligned). TMEM side: the transfer is cut into W = min(D/4, LANES)-word segments aligned in
// DRAM, one per cycle (CW/W per chunk), so the lanes need no shifter: lane l carries word l of
// the segment, and the partial segments at the ends are masked. The segment's words sit in
// consecutive TMEM words, hence distinct banks.
//   LD  the chunks are requested ahead into a DEPTH-chunk buffer (block RAM). A chunk's slot is
//       reserved when it is requested, so read data (which cannot be refused) always has room.
//       A segment leaves the buffer through its registered read and the TMEM write register.
//   ST  the segments read from TMEM are gathered into a chunk register; a chunk is written once,
//       word-masked, with its last segment in range (that segment comes straight from TMEM).
// The DRAM port may refuse a request (b_gnt low) and read data may take any time to return (in
// order). An ST completes once the DRAM has acknowledged all its writes (wr_idle).
module otpu_dma
  import otpu_pkg::*;
#(
  parameter int D     = 32,
  parameter int LANES = 8,
  parameter int DEPTH = 32                            // LD chunk buffer (a power of two)
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
  localparam int CW  = D / 4;                         // words per chunk
  localparam int W   = (CW < LANES) ? CW : LANES;     // words per segment
  localparam int SPC = CW / W;                        // segments per chunk
  localparam int SWL = $clog2(W);
  localparam int CWL = $clog2(CW);
  localparam int PW  = $clog2(DEPTH);
  initial if (DEPTH != (1 << PW) || DEPTH < 2) $fatal(1, "otpu_dma: DEPTH must be a power of two");

  logic        busy, is_st, ackw;
  assign rdy = !busy;
  logic [31:0] dw, de;                 // the DRAM word range [dw, de)
  // TMEM side: the segment at DRAM word address sw, its TMEM address, its lanes in range
  // (registered: the TMEM arbiter sees these lanes), and the segments left
  logic [31:0]  sw, so, sleft;
  logic [W-1:0] sm;

  function automatic logic in_rng(input logic [31:0] a);
    return (a >= dw) && (a < de);
  endfunction
  function automatic int pos_of(input logic [31:0] a);     // segment index within its chunk
    return int'((a % CW) / W);
  endfunction
  wire seg_end = (pos_of(sw) == SPC - 1) || (sleft == 1);   // the segment at sw ends its chunk

  // ---- LD: chunk requests (address ic, cleft left), the chunk buffer, segment delivery
  logic [31:0]   ic, cleft;
  logic [PW:0]   occ, cnt;             // chunks requested / received, and not yet delivered
  logic [PW-1:0] wp, rp;               // buffer slot of the next chunk received / delivered
  wire ld_act = busy && !is_st;
  wire ld_req = ld_act && (cleft != 0) && (occ != (PW+1)'(DEPTH));
  wire ld_iss = ld_req && b_gnt;
  wire ld_dv  = ld_act && (sleft != 0) && (cnt != 0);       // deliver the segment at sw
  wire ld_eat = ld_dv && seg_end;                           // ... which frees its chunk's slot

  // block RAM, written in its own reset-free process (see otpu_mxu); read every cycle
  (* ram_style = "block" *) logic [D*8-1:0] lb [DEPTH];
  logic [D*8-1:0] lb_q;
  always_ff @(posedge clk) begin
    if (b_rvalid) lb[wp] <= b_rdata;
    lb_q <= lb[rp];
  end
  // the delivered segment (in lb_q) goes to the TMEM write register; its position in the
  // chunk, lanes and TMEM address come with it (dv_*)
  logic                   dv_v, ld_last;
  logic [W-1:0]           dv_m;
  logic [31:0]            dv_a;
  int                     dv_p;
  logic [LANES-1:0]       lw_en;
  logic [LANES-1:0][31:0] lw_addr, lw_data;
  always_comb begin
    lw_en = '0; lw_addr = '0; lw_data = '0;
    for (int l = 0; l < W; l++) begin
      lw_en[l] = dv_v && dv_m[l];
      lw_addr[l] = dv_a + 32'(l);
      lw_data[l] = lb_q[32 * (dv_p * W + l) +: 32];
    end
  end
  always_ff @(posedge clk) begin
    t_wen <= rst ? '0 : lw_en;
    t_waddr <= lw_addr;
    t_wdata <= lw_data;
  end

  // ---- ST: the segment read last cycle, pending in t_rdata (its position, lanes, chunk, and
  // whether it ends its chunk), and the chunk gathered so far
  logic                   st_pend, pl;
  int                     pp;
  logic [W-1:0]           pm;
  logic [31:0]            pc;
  logic [CW-1:0][31:0]    cb;
  logic [CW-1:0]          cbm;
  wire st_wr = st_pend && pl;                       // the chunk's write request
  wire adv   = !st_wr || b_gnt;                     // the read -> write pipeline moves
  wire st_rd = busy && is_st && !ackw && (sleft != 0) && adv;

  always_comb begin
    b_req = ld_req; b_we = 1'b0; b_addr = ic;
    t_ren = '0; t_raddr = '0;
    for (int p = 0; p < SPC; p++)
      for (int l = 0; l < W; l++) begin
        b_wmask[p * W + l] = cbm[p * W + l] || (pp == p && pm[l]);
        b_wdata[32 * (p * W + l) +: 32] = (pp == p) ? t_rdata[l] : cb[p * W + l];
      end
    if (busy && is_st && !ackw) begin
      b_addr = pc;
      if (st_wr) begin
        b_req = 1'b1;
        b_we = 1'b1;
      end
      if (st_rd)
        for (int l = 0; l < W; l++) begin
          t_ren[l] = sm[l];
          t_raddr[l] = so + 32'(l);
        end
    end
  end

  logic ld_fin;                          // LD: the last write is in the write register
  always_ff @(posedge clk) begin
    done <= ld_fin;                      // an LD is done once its last TMEM write has landed
    ld_fin <= 1'b0;
    ld_last <= 1'b0;
    dv_v <= ld_dv;
    dv_m <= sm;
    dv_a <= so;
    dv_p <= pos_of(sw);
    if (rst) begin
      ld_fin <= 1'b0;
      dv_v <= 1'b0;
      busy <= 1'b0;
      st_pend <= 1'b0;
      ackw <= 1'b0;
    end else if (start) begin
      // one carry chain each: the counts from the start's offset in its segment / chunk
      logic [31:0] a, n, ow, oc;
      a = cmd.w1 >> 2;
      n = cmd.w3;
      ow = a % W;
      oc = a % CW;
      is_st <= (cmd.op == OP_ST);
      dw <= a;
      de <= a + n;
      sw <= a & ~32'(W - 1);
      so <= cmd.w2 - ow;
      for (int l = 0; l < W; l++) sm[l] <= (32'(l) >= ow) && (n > 32'(l) - ow);
      sleft <= (n + (ow + (W - 1))) >> SWL;
      ic <= a & ~32'(CW - 1);
      cleft <= (n + (oc + (CW - 1))) >> CWL;
      occ <= '0; cnt <= '0; wp <= '0; rp <= '0;
      st_pend <= 1'b0;
      cbm <= '0;
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
      if (ld_dv || st_rd) begin          // the TMEM side moves to the next segment
        sw <= sw + W;
        so <= so + W;
        sleft <= sleft - 1;
        for (int l = 0; l < W; l++) sm[l] <= in_rng(sw + W + 32'(l));
      end
      if (!is_st) begin
        if (ld_iss) begin
          ic <= ic + CW;
          cleft <= cleft - 1;
        end
        occ <= occ + (PW+1)'(ld_iss) - (PW+1)'(ld_eat);
        cnt <= cnt + (PW+1)'(b_rvalid) - (PW+1)'(ld_eat);
        if (b_rvalid) wp <= wp + 1'b1;
        if (ld_eat) rp <= rp + 1'b1;
        if (ld_dv && sleft == 1) ld_last <= 1'b1;
        if (ld_last) begin
          busy <= 1'b0;
          ld_fin <= 1'b1;
        end
      end else if (adv) begin
        if (st_pend) begin
          if (pl) cbm <= '0;                    // the chunk's write is taken this cycle
          else
            for (int l = 0; l < W; l++) begin
              cb[pp * W + l] <= t_rdata[l];
              cbm[pp * W + l] <= pm[l];
            end
        end
        st_pend <= st_rd;
        pp <= pos_of(sw);
        pm <= sm;
        pc <= sw & ~32'(CW - 1);
        pl <= seg_end;
        if (st_pend && sleft == 0) begin        // the last write is taken this cycle
          st_pend <= 1'b0;
          ackw <= 1'b1;
        end
      end
    end
  end
endmodule
