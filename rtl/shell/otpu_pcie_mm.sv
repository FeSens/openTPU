// LitePCIe's DMA streams onto the card's memory port (otpu_native_sys's x_* AXI4 slave, in place
// of XDMA's memory-mapped master): the PCIe core (boards/ypcb-00338/litepcie/otpu_litepcie.v,
// tools/litepcie/gen_core.py) moves host memory to and from two streams, and the driver queues
// card addresses beside them (docs/litepcie.md). One clock: the PCIe user clock (250 MHz).
//
// H2C (host -> card): a command {addr, len, flags} writes the next len bytes of the H2C stream at
// card byte address addr, in INCR bursts of 16-byte beats that never cross a 512-byte boundary
// (so never a 4 KiB one, nor the channels' bit 31: otpu_axi_split2). A burst's AW goes only once
// all of its beats are in the data FIFO, so W never waits on the stream and the channel is never
// held mid-burst; the stream is taken whenever the FIFO has room. h2c_done pulses once per
// command, when the last of its bursts' write responses is in (the data is then where every later
// access to it, the accelerator's included, finds it: otpu_mem_ch's B). Commands finish in order.
// flags[0] (FILL): write zeros, no stream data (the ECC scrub).
//
// C2H (card -> host): a command {addr, len} reads len bytes at addr onto the C2H stream (last on
// a command's final beat), in the same bursts. A burst's AR goes only with room for all of its
// beats in the read FIFO, so R is always taken whatever the stream does.
//
// addr and len are in bytes, multiples of 16 (the host's beats are 64 bytes); lower bits are
// dropped and flagged (status ERR_ALIGN). A command of no bytes finishes once every earlier one
// has. A response other than OKAY sets ERR_RESP. err: any error, sticky until rst.
//
// Timing (250 MHz beside the accelerator): both streams and the AXI master pass register slices
// (otpu_skid, otpu_axi_slice), so every path to LitePCIe and to the memory port, which sit apart
// on the die, runs flip-flop to flip-flop. A burst's length is registered a cycle after its
// command position moves (a burst issues at most every other cycle; bursts are up to 32 beats),
// and the FIFOs' room and promised beats are counters, so no burst decision subtracts pointers.
//
// status: [0] H2C busy, [1] C2H busy, [8:4] write bursts in flight, [16] ERR_ALIGN,
// [17] ERR_BRESP, [18] ERR_RRESP.
module otpu_pcie_mm #(
  parameter int IDW = 4,
  parameter int HD  = 256,        // H2C data FIFO, beats (>= 2 bursts)
  parameter int CD  = 256,        // C2H read FIFO, beats (>= 2 bursts): the reads in flight
  parameter int WO  = 16          // write bursts in flight
) (
  input  logic            clk,
  input  logic            rst,
  // H2C: commands and the stream (LitePCIe's DMA reader)
  input  logic            h2c_cmd_valid,
  output logic            h2c_cmd_ready,
  input  logic [31:0]     h2c_cmd_addr,
  input  logic [23:0]     h2c_cmd_len,
  input  logic [7:0]      h2c_cmd_flags,
  input  logic            h2c_dat_valid,
  output logic            h2c_dat_ready,
  input  logic [127:0]    h2c_dat_data,
  output logic            h2c_done,
  // C2H: commands and the stream (LitePCIe's DMA writer)
  input  logic            c2h_cmd_valid,
  output logic            c2h_cmd_ready,
  input  logic [31:0]     c2h_cmd_addr,
  input  logic [23:0]     c2h_cmd_len,
  input  logic [7:0]      c2h_cmd_flags,
  output logic            c2h_dat_valid,
  input  logic            c2h_dat_ready,
  output logic [127:0]    c2h_dat_data,
  output logic            c2h_dat_last,
  output logic [31:0]     status,
  output logic            err,
  // AXI4 master (otpu_native_sys x_*): INCR bursts of 16-byte beats, ID 0
  output logic            m_awvalid,
  input  logic            m_awready,
  output logic [IDW-1:0]  m_awid,
  output logic [31:0]     m_awaddr,
  output logic [7:0]      m_awlen,
  output logic            m_wvalid,
  input  logic            m_wready,
  output logic [127:0]    m_wdata,
  output logic [15:0]     m_wstrb,
  output logic            m_wlast,
  input  logic            m_bvalid,
  output logic            m_bready,
  input  logic [IDW-1:0]  m_bid,
  input  logic [1:0]      m_bresp,
  output logic            m_arvalid,
  input  logic            m_arready,
  output logic [IDW-1:0]  m_arid,
  output logic [31:0]     m_araddr,
  output logic [7:0]      m_arlen,
  input  logic            m_rvalid,
  output logic            m_rready,
  input  logic [IDW-1:0]  m_rid,
  input  logic [127:0]    m_rdata,
  input  logic [1:0]      m_rresp,
  input  logic            m_rlast
);
  localparam int BB = 32;                         // beats a burst at most (512 bytes)
  localparam int HAW = $clog2(HD);
  localparam int CAW = $clog2(CD);

  logic err_align_h, err_align_c, err_bresp, err_rresp;

  // beats from addr to the end of its 512-byte block, at most rem
  function automatic logic [5:0] burst_len(input logic [27:0] beat, input logic [19:0] rem);
    logic [5:0] room;
    room = 6'(BB) - {1'b0, beat[4:0]};
    return (20'(room) < rem) ? room : rem[5:0];
  endfunction

  // ------------------------------------------------------------------ the AXI master's slice
  logic         a_awvalid, a_awready, a_wvalid, a_wready, a_wlast, a_bvalid;
  logic         a_arvalid, a_arready, a_rvalid, a_rlast;
  logic [31:0]  a_awaddr, a_araddr;
  logic [7:0]   a_awlen, a_arlen;
  logic [127:0] a_wdata, a_rdata;
  logic [1:0]   a_bresp, a_rresp;
  otpu_axi_slice #(.REG(1'b1), .IDW(IDW), .DW(128)) u_ms (.clk, .rst,
    .s_awvalid(a_awvalid), .s_awready(a_awready), .s_awid('0), .s_awaddr(a_awaddr),
    .s_awlen(a_awlen), .s_wvalid(a_wvalid), .s_wready(a_wready), .s_wdata(a_wdata), .s_wstrb('1),
    .s_wlast(a_wlast), .s_bvalid(a_bvalid), .s_bready(1'b1), .s_bid(), .s_bresp(a_bresp),
    .s_arvalid(a_arvalid), .s_arready(a_arready), .s_arid('0), .s_araddr(a_araddr),
    .s_arlen(a_arlen), .s_rvalid(a_rvalid), .s_rready(1'b1), .s_rid(), .s_rdata(a_rdata),
    .s_rresp(a_rresp), .s_rlast(a_rlast),
    .m_awvalid, .m_awready, .m_awid, .m_awaddr, .m_awlen, .m_wvalid, .m_wready, .m_wdata,
    .m_wstrb, .m_wlast, .m_bvalid, .m_bready, .m_bid, .m_bresp, .m_arvalid, .m_arready, .m_arid,
    .m_araddr, .m_arlen, .m_rvalid, .m_rready, .m_rid, .m_rdata, .m_rresp, .m_rlast);

  // ------------------------------------------------------------------ H2C
  // the stream through a slice, into the data FIFO: show-ahead, a RAM with a registered read (hq:
  // the head). h_room: the RAM takes a beat; from a count two cycles old, so it holds 2 back
  logic         hs_valid, hs_ready;
  logic [127:0] hs_data;
  otpu_skid #(.W(128)) u_hs (.clk, .rst, .s_valid(h2c_dat_valid), .s_ready(h2c_dat_ready),
    .s_data(h2c_dat_data), .m_valid(hs_valid), .m_ready(hs_ready), .m_data(hs_data));
  logic [127:0] hmem [HD];
  logic [HAW-1:0] hwp, hrp;
  logic [HAW:0] h_ram;                            // entries in the RAM
  logic [127:0] hq;
  logic         hq_v, h_room;
  logic         h_push, h_pop, h_load;
  assign hs_ready = h_room;
  assign h_push = hs_valid && h_room;
  assign h_load = h_ram != 0 && (!hq_v || h_pop);
  always_ff @(posedge clk) begin
    if (h_push) hmem[hwp] <= hs_data;
    if (h_load) hq <= hmem[hrp];
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      hwp <= '0; hrp <= '0; h_ram <= '0; hq_v <= 1'b0; h_room <= 1'b0;
    end else begin
      if (h_push) hwp <= hwp + 1'b1;
      if (h_load) hrp <= hrp + 1'b1;
      h_ram <= h_ram + (HAW+1)'(h_push) - (HAW+1)'(h_load);
      h_room <= h_ram <= (HAW+1)'(HD - 3);
      if (h_load) hq_v <= 1'b1;
      else if (h_pop) hq_v <= 1'b0;
    end
  end

  // the command being cut into bursts; hb_*: its next burst, registered (hb_ok) a cycle after
  // the position moves
  logic        hc_act, hc_fill;
  logic [27:0] hc_beat;                           // next beat's card address / 16
  logic [19:0] hc_rem;                            // beats left to issue
  logic [5:0]  hb_len;
  logic        hb_ok, hb_last;
  logic [HAW+1:0] h_avail;                        // FIFO beats (pushed) no burst has promised
  logic [$clog2(WO+1)-1:0] h_out;                 // bursts issued, response not in
  logic        aw_go, h_zero, h_take;
  // the W queue (bursts issued, W not sent) and the B queue (bursts whose response is awaited):
  // {last burst of its command, FILL, beats}
  localparam int QW = 1 + 1 + 6;
  logic [QW-1:0] wq [WO];
  logic [QW-1:0] bq [WO];
  logic [$clog2(WO):0] wq_wp, wq_rp, bq_wp, bq_rp;
  logic        wq_empty, bq_empty;
  assign wq_empty = wq_wp == wq_rp;
  assign bq_empty = bq_wp == bq_rp;
  // a burst issues with its beats in the FIFO (or FILL) and a free response slot
  assign aw_go = hb_ok && !a_awvalid && h_out != ($clog2(WO+1))'(WO) &&
                 (hc_fill || h_avail >= (HAW+2)'(hb_len));
  // a command of no bytes: finishes when every earlier burst has its response
  assign h_zero = hc_act && hc_rem == 0;
  assign h2c_cmd_ready = !hc_act;
  assign h_take = h2c_cmd_valid && !hc_act;

  // W: the head of the W queue, beat by beat
  logic [5:0] w_cnt;                              // beats of the head burst sent
  logic       w_fill, w_last_cmd;
  logic [5:0] w_len;
  assign {w_last_cmd, w_fill, w_len} = wq[wq_rp[$clog2(WO)-1:0]];
  assign a_wvalid = !wq_empty && (w_fill || hq_v);
  assign a_wdata = w_fill ? '0 : hq;
  assign a_wlast = w_cnt == w_len - 1'b1;
  assign h_pop = a_wvalid && a_wready && !w_fill;

  logic h_done_r;
  always_ff @(posedge clk) begin
    if (aw_go) begin
      wq[wq_wp[$clog2(WO)-1:0]] <= {hb_last, hc_fill, hb_len};
      bq[bq_wp[$clog2(WO)-1:0]] <= {hb_last, hc_fill, hb_len};
    end
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      hc_act <= 1'b0; hc_fill <= 1'b0; hc_beat <= '0; hc_rem <= '0;
      hb_len <= '0; hb_ok <= 1'b0; hb_last <= 1'b0;
      a_awvalid <= 1'b0; a_awaddr <= '0; a_awlen <= '0;
      wq_wp <= '0; wq_rp <= '0; bq_wp <= '0; bq_rp <= '0;
      w_cnt <= '0; h_avail <= '0; h_out <= '0; h_done_r <= 1'b0;
      err_align_h <= 1'b0; err_bresp <= 1'b0;
    end else begin
      h_done_r <= 1'b0;
      // take a command
      if (h_take) begin
        hc_act <= 1'b1;
        hc_fill <= h2c_cmd_flags[0];
        hc_beat <= h2c_cmd_addr[31:4];
        hc_rem <= h2c_cmd_len[23:4];
        if (h2c_cmd_addr[3:0] != 0 || h2c_cmd_len[3:0] != 0) err_align_h <= 1'b1;
      end
      // the next burst, from the position as it stands
      if (h_take || aw_go) hb_ok <= 1'b0;
      else begin
        hb_len <= burst_len(hc_beat, hc_rem);
        hb_last <= 20'(burst_len(hc_beat, hc_rem)) == hc_rem;
        hb_ok <= hc_act && hc_rem != 0;
      end
      // issue a burst
      if (aw_go) begin
        a_awvalid <= 1'b1;
        a_awaddr <= {hc_beat, 4'h0};
        a_awlen <= 8'(hb_len - 1'b1);
        hc_beat <= hc_beat + 28'(hb_len);
        hc_rem <= hc_rem - 20'(hb_len);
        if (hb_last) hc_act <= 1'b0;
        wq_wp <= wq_wp + 1'b1;
        bq_wp <= bq_wp + 1'b1;
      end else if (h_zero && bq_empty) begin
        hc_act <= 1'b0;
        h_done_r <= 1'b1;
      end
      if (a_awvalid && a_awready) a_awvalid <= 1'b0;
      // W beats
      if (a_wvalid && a_wready) begin
        if (a_wlast) begin
          w_cnt <= '0;
          wq_rp <= wq_rp + 1'b1;
        end else begin
          w_cnt <= w_cnt + 1'b1;
        end
      end
      // the FIFO's unpromised beats: up by one a beat pushed, down by a burst's at its AW
      h_avail <= h_avail + (HAW+2)'(h_push) - ((aw_go && !hc_fill) ? (HAW+2)'(hb_len) : '0);
      h_out <= h_out + ($clog2(WO+1))'(aw_go) - ($clog2(WO+1))'(a_bvalid && !bq_empty);
      // responses, in burst order (one ID)
      if (a_bvalid && !bq_empty) begin
        bq_rp <= bq_rp + 1'b1;
        if (bq[bq_rp[$clog2(WO)-1:0]][QW-1]) h_done_r <= 1'b1;
        if (a_bresp != 2'b00) err_bresp <= 1'b1;
      end
    end
  end
  assign h2c_done = h_done_r;

  // ------------------------------------------------------------------ C2H
  // the read FIFO: show-ahead, {last, data}, out through a slice (cs_ready: it takes the head)
  logic [128:0] cmem [CD];
  logic [CAW-1:0] cwp, crp;
  logic [CAW:0] c_ram;
  logic [128:0] cq;
  logic         cq_v, cs_ready;
  logic         c_push, c_pop, c_load;
  assign c_pop = cq_v && cs_ready;
  assign c_load = c_ram != 0 && (!cq_v || c_pop);
  otpu_skid #(.W(129)) u_cs (.clk, .rst, .s_valid(cq_v), .s_ready(cs_ready), .s_data(cq),
    .m_valid(c2h_dat_valid), .m_ready(c2h_dat_ready), .m_data({c2h_dat_last, c2h_dat_data}));

  logic        cc_act;
  logic [27:0] cc_beat;
  logic [19:0] cc_rem;
  logic [5:0]  cb_len;
  logic        cb_ok, cb_last, c_take;
  logic [CAW+1:0] c_free;                         // FIFO entries (RAM and head) no burst has promised
  logic        ar_go;
  // the R queue: {last burst of its command, beats}
  logic [6:0] rq [16];
  logic [4:0] rq_wp, rq_rp, rq_n;
  logic [5:0] r_cnt;
  logic       rq_full, rq_empty;
  assign rq_n = rq_wp - rq_rp;
  assign rq_full = rq_n == 5'd16;
  assign rq_empty = rq_wp == rq_rp;
  assign ar_go = cb_ok && !a_arvalid && !rq_full && c_free >= (CAW+2)'(cb_len);
  assign c2h_cmd_ready = !cc_act;
  assign c_take = c2h_cmd_valid && !cc_act;
  logic r_lastcmd;
  logic [5:0] r_len;
  assign {r_lastcmd, r_len} = rq[rq_rp[3:0]];
  assign c_push = a_rvalid;

  always_ff @(posedge clk) begin
    if (c_push) cmem[cwp] <= {r_lastcmd && r_cnt == r_len - 1'b1, a_rdata};
    if (c_load) cq <= cmem[crp];
    if (ar_go) rq[rq_wp[3:0]] <= {cb_last, cb_len};
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      cwp <= '0; crp <= '0; c_ram <= '0; cq_v <= 1'b0;
      cc_act <= 1'b0; cc_beat <= '0; cc_rem <= '0;
      cb_len <= '0; cb_ok <= 1'b0; cb_last <= 1'b0;
      a_arvalid <= 1'b0; a_araddr <= '0; a_arlen <= '0;
      rq_wp <= '0; rq_rp <= '0; r_cnt <= '0; c_free <= (CAW+2)'(CD);
      err_rresp <= 1'b0; err_align_c <= 1'b0;
    end else begin
      if (c_push) cwp <= cwp + 1'b1;
      if (c_load) crp <= crp + 1'b1;
      c_ram <= c_ram + (CAW+1)'(c_push) - (CAW+1)'(c_load);
      if (c_load) cq_v <= 1'b1;
      else if (c_pop) cq_v <= 1'b0;
      if (c_take) begin
        cc_beat <= c2h_cmd_addr[31:4];
        cc_rem <= c2h_cmd_len[23:4];
        cc_act <= c2h_cmd_len[23:4] != 0;
        if (c2h_cmd_addr[3:0] != 0 || c2h_cmd_len[3:0] != 0) err_align_c <= 1'b1;
      end
      if (c_take || ar_go) cb_ok <= 1'b0;
      else begin
        cb_len <= burst_len(cc_beat, cc_rem);
        cb_last <= 20'(burst_len(cc_beat, cc_rem)) == cc_rem;
        cb_ok <= cc_act && cc_rem != 0;
      end
      if (ar_go) begin
        a_arvalid <= 1'b1;
        a_araddr <= {cc_beat, 4'h0};
        a_arlen <= 8'(cb_len - 1'b1);
        cc_beat <= cc_beat + 28'(cb_len);
        cc_rem <= cc_rem - 20'(cb_len);
        if (cb_last) cc_act <= 1'b0;
        rq_wp <= rq_wp + 1'b1;
      end
      if (a_arvalid && a_arready) a_arvalid <= 1'b0;
      // the FIFO's unpromised room: down by a burst's at its AR, up by one a beat out (a beat in
      // fills room its burst promised)
      c_free <= c_free - (ar_go ? (CAW+2)'(cb_len) : '0) + (CAW+2)'(c_pop);
      if (c_push) begin
        if (r_cnt == r_len - 1'b1) begin
          r_cnt <= '0;
          rq_rp <= rq_rp + 1'b1;
        end else begin
          r_cnt <= r_cnt + 1'b1;
        end
        if (a_rresp != 2'b00) err_rresp <= 1'b1;
      end
    end
  end

  // ------------------------------------------------------------------ status
  logic [4:0] h_out5;
  assign h_out5 = 5'(h_out);
  logic err_align;
  assign err_align = err_align_h || err_align_c;
  assign status = {13'd0, err_rresp, err_bresp, err_align, 7'd0, h_out5, 2'd0,
                   cc_act || cq_v || !rq_empty || c2h_dat_valid, hc_act || !bq_empty};
  assign err = err_align || err_bresp || err_rresp;

`ifndef SYNTHESIS
  // the AXI master's promises: a burst's beats never cross 512 bytes, W never starves a promised
  // burst, R never finds a full FIFO, the FIFOs never overflow
  always_ff @(posedge clk) if (!rst) begin
    if (a_awvalid && (a_awaddr[8:4] + a_awlen) > 31)
      $fatal(1, "otpu_pcie_mm: write burst at %h len %0d crosses 512 bytes", a_awaddr, a_awlen);
    if (a_arvalid && (a_araddr[8:4] + a_arlen) > 31)
      $fatal(1, "otpu_pcie_mm: read burst at %h len %0d crosses 512 bytes", a_araddr, a_arlen);
    if (c_push && c_ram == (CAW+1)'(CD))
      $fatal(1, "otpu_pcie_mm: read data with the read FIFO full");
    if (h_push && h_ram == (HAW+1)'(HD))
      $fatal(1, "otpu_pcie_mm: stream data with the data FIFO full");
    if (c_push && rq_empty)
      $fatal(1, "otpu_pcie_mm: read data with no read burst outstanding");
    if (a_bvalid && bq_empty)
      $fatal(1, "otpu_pcie_mm: write response with no write burst outstanding");
    if (h_avail > (HAW+2)'(HD + 1))
      $fatal(1, "otpu_pcie_mm: unpromised beats %0d past the FIFO", h_avail);
    // a burst's beats are in the FIFO from its AW on: W never waits for the stream
    if (!wq_empty && !w_fill && !hq_v && $past(!wq_empty))
      $fatal(1, "otpu_pcie_mm: W waits for stream data in a promised burst (avail %0d ram %0d wq %0d w_cnt %0d w_len %0d)",
             h_avail, h_ram, wq_wp - wq_rp, w_cnt, w_len);
  end
`endif
endmodule
