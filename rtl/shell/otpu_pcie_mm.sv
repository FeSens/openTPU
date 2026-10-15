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
// beats in the read FIFO, so R is always taken (rready high) whatever the stream does.
//
// addr and len are in bytes, multiples of 16 (the host's beats are 64 bytes); lower bits are
// dropped and flagged (status ERR_ALIGN). A command of no bytes finishes once every earlier one
// has. A response other than OKAY sets ERR_RESP. err: any error, sticky until rst.
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

  // ------------------------------------------------------------------ H2C
  // the data FIFO: show-ahead, a RAM with a registered read (hq: the head)
  logic [127:0] hmem [HD];
  logic [HAW:0] hwp, hrp;                         // RAM pointers (hq not counted)
  logic [127:0] hq;
  logic         hq_v;
  logic         h_push, h_pop, h_load;
  logic [HAW:0]   h_ram;                          // entries in the RAM (pointer width: they wrap)
  logic [HAW+1:0] h_count;                        // entries, hq included
  assign h_ram = hwp - hrp;
  assign h_count = {1'b0, h_ram} + (HAW+2)'(hq_v);
  assign h_push = h2c_dat_valid && h2c_dat_ready;
  assign h_load = h_ram != 0 && (!hq_v || h_pop);
  assign h2c_dat_ready = h_ram != (HAW+1)'(HD);
  always_ff @(posedge clk) begin
    if (h_push) hmem[hwp[HAW-1:0]] <= h2c_dat_data;
    if (h_load) hq <= hmem[hrp[HAW-1:0]];
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      hwp <= '0; hrp <= '0; hq_v <= 1'b0;
    end else begin
      if (h_push) hwp <= hwp + 1'b1;
      if (h_load) begin hrp <= hrp + 1'b1; hq_v <= 1'b1; end
      else if (h_pop) hq_v <= 1'b0;
    end
  end

  // the command being cut into bursts
  logic        hc_act, hc_fill;
  logic [27:0] hc_beat;                           // next beat's card address / 16
  logic [19:0] hc_rem;                            // beats left to issue
  logic [5:0]  hb_len;
  logic [HAW+1:0] h_res;                          // FIFO beats promised to issued bursts, not yet sent
  logic [$clog2(WO+1)-1:0] h_out;                 // bursts issued, response not in
  logic        aw_go, h_zero;
  assign hb_len = burst_len(hc_beat, hc_rem);
  // the W queue (bursts issued, W not sent) and the B queue (bursts whose response is awaited):
  // {last burst of its command, FILL, beats}
  localparam int QW = 1 + 1 + 6;
  logic [QW-1:0] wq [WO];
  logic [QW-1:0] bq [WO];
  logic [$clog2(WO):0] wq_wp, wq_rp, bq_wp, bq_rp;
  logic        wq_empty, bq_empty;
  assign wq_empty = wq_wp == wq_rp;
  assign bq_empty = bq_wp == bq_rp;
  logic        last_burst;
  assign last_burst = 20'(hb_len) == hc_rem;
  // a burst issues with its beats in the FIFO (or FILL) and a free response slot
  assign aw_go = hc_act && hc_rem != 0 && !m_awvalid && h_out != ($clog2(WO+1))'(WO) &&
                 (hc_fill || (h_count - h_res) >= (HAW+2)'(hb_len));
  // a command of no bytes: finishes when every earlier burst has its response
  assign h_zero = hc_act && hc_rem == 0;
  assign h2c_cmd_ready = !hc_act;

  // W: the head of the W queue, beat by beat
  logic [5:0] w_cnt;                              // beats of the head burst sent
  logic       w_fill, w_last_cmd;
  logic [5:0] w_len;
  assign {w_last_cmd, w_fill, w_len} = wq[wq_rp[$clog2(WO)-1:0]];
  assign m_wvalid = !wq_empty && (w_fill || hq_v);
  assign m_wdata = w_fill ? '0 : hq;
  assign m_wstrb = '1;
  assign m_wlast = w_cnt == w_len - 1'b1;
  assign h_pop = m_wvalid && m_wready && !w_fill;
  assign m_awid = '0;
  assign m_bready = 1'b1;

  logic h_done_r;
  always_ff @(posedge clk) begin
    if (rst) begin
      hc_act <= 1'b0; hc_fill <= 1'b0; hc_beat <= '0; hc_rem <= '0;
      m_awvalid <= 1'b0; m_awaddr <= '0; m_awlen <= '0;
      wq_wp <= '0; wq_rp <= '0; bq_wp <= '0; bq_rp <= '0;
      w_cnt <= '0; h_res <= '0; h_out <= '0; h_done_r <= 1'b0;
      err_align_h <= 1'b0; err_bresp <= 1'b0;
    end else begin
      h_done_r <= 1'b0;
      // take a command
      if (h2c_cmd_valid && h2c_cmd_ready) begin
        hc_act <= 1'b1;
        hc_fill <= h2c_cmd_flags[0];
        hc_beat <= h2c_cmd_addr[31:4];
        hc_rem <= h2c_cmd_len[23:4];
        if (h2c_cmd_addr[3:0] != 0 || h2c_cmd_len[3:0] != 0) err_align_h <= 1'b1;
      end
      // issue a burst
      if (aw_go) begin
        m_awvalid <= 1'b1;
        m_awaddr <= {hc_beat, 4'h0};
        m_awlen <= 8'(hb_len - 1'b1);
        hc_beat <= hc_beat + 28'(hb_len);
        hc_rem <= hc_rem - 20'(hb_len);
        if (last_burst) hc_act <= 1'b0;
        wq[wq_wp[$clog2(WO)-1:0]] <= {last_burst, hc_fill, hb_len};
        bq[bq_wp[$clog2(WO)-1:0]] <= {last_burst, hc_fill, hb_len};
        wq_wp <= wq_wp + 1'b1;
        bq_wp <= bq_wp + 1'b1;
      end else if (h_zero && bq_empty) begin
        hc_act <= 1'b0;
        h_done_r <= 1'b1;
      end
      if (m_awvalid && m_awready) m_awvalid <= 1'b0;
      // W beats
      if (m_wvalid && m_wready) begin
        if (m_wlast) begin
          w_cnt <= '0;
          wq_rp <= wq_rp + 1'b1;
        end else begin
          w_cnt <= w_cnt + 1'b1;
        end
      end
      // the beats promised to bursts: up by a burst's at its AW, down by one a W beat sent
      h_res <= h_res + ((aw_go && !hc_fill) ? (HAW+2)'(hb_len) : '0) - (HAW+2)'(h_pop);
      h_out <= h_out + ($clog2(WO+1))'(aw_go) - ($clog2(WO+1))'(m_bvalid && m_bready && !bq_empty);
      // responses, in burst order (one ID)
      if (m_bvalid && m_bready && !bq_empty) begin
        bq_rp <= bq_rp + 1'b1;
        if (bq[bq_rp[$clog2(WO)-1:0]][QW-1]) h_done_r <= 1'b1;
        if (m_bresp != 2'b00) err_bresp <= 1'b1;
      end
    end
  end
  assign h2c_done = h_done_r;

  // ------------------------------------------------------------------ C2H
  // the read FIFO: show-ahead, {last, data}
  logic [128:0] cmem [CD];
  logic [CAW:0] cwp, crp;
  logic [128:0] cq;
  logic         cq_v;
  logic         c_push, c_pop, c_load;
  logic [CAW:0]   c_ram;
  logic [CAW+1:0] c_count;
  assign c_ram = cwp - crp;
  assign c_count = {1'b0, c_ram} + (CAW+2)'(cq_v);
  assign c_load = c_ram != 0 && (!cq_v || c_pop);
  assign c_pop = c2h_dat_valid && c2h_dat_ready;
  assign c2h_dat_valid = cq_v;
  assign {c2h_dat_last, c2h_dat_data} = cq;

  logic        cc_act;
  logic [27:0] cc_beat;
  logic [19:0] cc_rem;
  logic [5:0]  cb_len;
  logic [CAW+1:0] c_res;                          // FIFO room promised to issued bursts, not yet filled
  logic        ar_go;
  assign cb_len = burst_len(cc_beat, cc_rem);
  // the R queue: {last burst of its command, beats}
  logic [6:0] rq [16];
  logic [4:0] rq_wp, rq_rp, rq_n;
  logic [5:0] r_cnt;
  logic       rq_full, rq_empty;
  assign rq_n = rq_wp - rq_rp;
  assign rq_full = rq_n == 5'd16;
  assign rq_empty = rq_wp == rq_rp;
  assign ar_go = cc_act && cc_rem != 0 && !m_arvalid && !rq_full &&
                 ((CAW+2)'(CD) - c_count - c_res) >= (CAW+2)'(cb_len);
  assign c2h_cmd_ready = !cc_act;
  assign m_arid = '0;
  assign m_rready = 1'b1;
  logic r_lastcmd;
  logic [5:0] r_len;
  assign {r_lastcmd, r_len} = rq[rq_rp[3:0]];
  assign c_push = m_rvalid && m_rready;

  always_ff @(posedge clk) begin
    if (c_push) cmem[cwp[CAW-1:0]] <= {r_lastcmd && r_cnt == r_len - 1'b1, m_rdata};
    if (c_load) cq <= cmem[crp[CAW-1:0]];
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      cwp <= '0; crp <= '0; cq_v <= 1'b0;
      cc_act <= 1'b0; cc_beat <= '0; cc_rem <= '0;
      m_arvalid <= 1'b0; m_araddr <= '0; m_arlen <= '0;
      rq_wp <= '0; rq_rp <= '0; r_cnt <= '0; c_res <= '0;
      err_rresp <= 1'b0; err_align_c <= 1'b0;
    end else begin
      if (c_push) cwp <= cwp + 1'b1;
      if (c_load) begin crp <= crp + 1'b1; cq_v <= 1'b1; end
      else if (c_pop) cq_v <= 1'b0;
      if (c2h_cmd_valid && c2h_cmd_ready) begin
        cc_beat <= c2h_cmd_addr[31:4];
        cc_rem <= c2h_cmd_len[23:4];
        cc_act <= c2h_cmd_len[23:4] != 0;
        if (c2h_cmd_addr[3:0] != 0 || c2h_cmd_len[3:0] != 0) err_align_c <= 1'b1;
      end
      if (ar_go) begin
        m_arvalid <= 1'b1;
        m_araddr <= {cc_beat, 4'h0};
        m_arlen <= 8'(cb_len - 1'b1);
        cc_beat <= cc_beat + 28'(cb_len);
        cc_rem <= cc_rem - 20'(cb_len);
        if (20'(cb_len) == cc_rem) cc_act <= 1'b0;
        rq[rq_wp[3:0]] <= {20'(cb_len) == cc_rem, cb_len};
        rq_wp <= rq_wp + 1'b1;
      end
      if (m_arvalid && m_arready) m_arvalid <= 1'b0;
      c_res <= c_res + (ar_go ? (CAW+2)'(cb_len) : '0) - (CAW+2)'(c_push);
      if (c_push) begin
        if (r_cnt == r_len - 1'b1) begin
          r_cnt <= '0;
          rq_rp <= rq_rp + 1'b1;
        end else begin
          r_cnt <= r_cnt + 1'b1;
        end
        if (m_rresp != 2'b00) err_rresp <= 1'b1;
      end
    end
  end

  // ------------------------------------------------------------------ status
  logic [4:0] h_out5;
  assign h_out5 = 5'(h_out);
  logic err_align;
  assign err_align = err_align_h || err_align_c;
  assign status = {13'd0, err_rresp, err_bresp, err_align, 7'd0, h_out5, 2'd0, cc_act || cq_v || !rq_empty,
                   hc_act || !bq_empty};
  assign err = err_align || err_bresp || err_rresp;

`ifndef SYNTHESIS
  // the AXI master's promises: a burst's beats never cross 512 bytes, W never starves a promised
  // burst, R never finds a full FIFO
  always_ff @(posedge clk) if (!rst) begin
    if (m_awvalid && (m_awaddr[8:4] + m_awlen) > 31)
      $fatal(1, "otpu_pcie_mm: write burst at %h len %0d crosses 512 bytes", m_awaddr, m_awlen);
    if (m_arvalid && (m_araddr[8:4] + m_arlen) > 31)
      $fatal(1, "otpu_pcie_mm: read burst at %h len %0d crosses 512 bytes", m_araddr, m_arlen);
    if (c_push && c_ram == (CAW+1)'(CD))
      $fatal(1, "otpu_pcie_mm: read data with the read FIFO full");
    if (c_push && rq_empty)
      $fatal(1, "otpu_pcie_mm: read data with no read burst outstanding");
    if (m_bvalid && bq_empty)
      $fatal(1, "otpu_pcie_mm: write response with no write burst outstanding");
    // a burst's beats are in the FIFO from its AW on: W never waits for the stream
    if (!wq_empty && !w_fill && !hq_v && $past(!wq_empty))
      $fatal(1, "otpu_pcie_mm: W waits for stream data in a promised burst (count %0d res %0d wq %0d w_cnt %0d w_len %0d hwp %0d hrp %0d)",
             h_count, h_res, wq_wp - wq_rp, w_cnt, w_len, hwp, hrp);
  end
`endif
endmodule
