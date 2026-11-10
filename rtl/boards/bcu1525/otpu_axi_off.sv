// AXI4 (no IDs: in-order responses) in front of a slave that can be switched off: the BCU1525's
// full top puts one between XDMA's DMA master for a channel (the block design's sc_dma, xdma_aclk)
// and the channel's bridge (otpu_mem_ch), whose controller the governor turns off (otpu_bcu_gov:
// the DIMM's LiteDRAM core in reset, its clocks stopped). With the controller's clock stopped the
// bridge answers nothing, and a reset of XDMA's side is not even acknowledged (its hold crosses
// through the controller's clock), so without this gate XDMA would wait for ever.
//   off:  no new burst for the slave. Bursts it has (commands taken, responses not all returned)
//         finish there; once it has none, the gate answers every burst itself: reads with ARLEN + 1
//         beats of SLVERR (data RDATA_OFF), writes by taking their data and answering SLVERR.
//   dead: the slave answers nothing from now (its clock has stopped or will): the gate answers
//         the bursts the slave had too, in their order, taking what is left of their write data,
//         and takes and drops whatever the slave still returns (it is reset before it is on again).
// The gate goes back to the slave once neither holds and nothing it answers is outstanding (it
// takes no new burst meanwhile, so none it took can reach the slave's responses). The
// owner raises off a while before dead (the governor's dimm_ok, OFF_DLY cycles ahead of dimm_on),
// so bursts in flight usually finish on the slave. NOUT bursts each way at most are outstanding;
// a read burst's length is kept for each, so a dead slave's reads can be completed here. A beat the
// slave presents when dead rises is replaced by the gate's (an emergency: data changes under valid).
module otpu_axi_off #(
  parameter int AW   = 31,
  parameter int DW   = 128,
  parameter int NOUT = 16,
  parameter logic [31:0] RDATA_OFF = 32'hDEAD_0FF0
) (
  input  logic            clk,
  input  logic            rst,
  input  logic            off,
  input  logic            dead,
  // the master's side
  input  logic [AW-1:0]   m_awaddr,
  input  logic [7:0]      m_awlen,
  input  logic            m_awvalid,
  output logic            m_awready,
  input  logic [DW-1:0]   m_wdata,
  input  logic [DW/8-1:0] m_wstrb,
  input  logic            m_wlast,
  input  logic            m_wvalid,
  output logic            m_wready,
  output logic [1:0]      m_bresp,
  output logic            m_bvalid,
  input  logic            m_bready,
  input  logic [AW-1:0]   m_araddr,
  input  logic [7:0]      m_arlen,
  input  logic            m_arvalid,
  output logic            m_arready,
  output logic [DW-1:0]   m_rdata,
  output logic [1:0]      m_rresp,
  output logic            m_rlast,
  output logic            m_rvalid,
  input  logic            m_rready,
  // the slave's side
  output logic [AW-1:0]   s_awaddr,
  output logic [7:0]      s_awlen,
  output logic            s_awvalid,
  input  logic            s_awready,
  output logic [DW-1:0]   s_wdata,
  output logic [DW/8-1:0] s_wstrb,
  output logic            s_wlast,
  output logic            s_wvalid,
  input  logic            s_wready,
  input  logic [1:0]      s_bresp,
  input  logic            s_bvalid,
  output logic            s_bready,
  output logic [AW-1:0]   s_araddr,
  output logic [7:0]      s_arlen,
  output logic            s_arvalid,
  input  logic            s_arready,
  input  logic [DW-1:0]   s_rdata,
  input  logic [1:0]      s_rresp,
  input  logic            s_rlast,
  input  logic            s_rvalid,
  output logic            s_rready
);
  localparam int CW = $clog2(NOUT) + 2;     // counts; n_wl may run ahead of n_aw (W before AW)
  localparam int PW = $clog2(NOUT);

  // ---------------------------------------------------------------- writes
  // n_aw: bursts whose AW is in (the slave's or here) and whose B the master has not taken; n_wl:
  // the same for WLAST; w_mid: a burst's data part way in. A burst's B can be given once its AW and
  // WLAST are in. wg: the gate answers writes.
  logic          wg, w_mid;
  logic [CW-1:0] n_aw, n_wl;
  wire w_here  = wg || dead;
  wire w_clean = n_aw == '0 && n_wl == '0 && !w_mid;
  wire aw_room = n_aw < CW'(NOUT);
  // an AW whose data has gone (or started) to the slave goes there too, off or not: the slave's W
  // channel would otherwise keep beats of a burst it never sees
  wire aw_fwd  = !w_here && aw_room && (!off || n_wl + CW'(w_mid) > n_aw);
  assign s_awaddr  = m_awaddr;
  assign s_awlen   = m_awlen;
  assign s_awvalid = aw_fwd && m_awvalid;
  assign m_awready = w_here ? aw_room && (off || dead) : aw_fwd && s_awready;
  assign s_wdata   = m_wdata;
  assign s_wstrb   = m_wstrb;
  assign s_wlast   = m_wlast;
  assign s_wvalid  = !w_here && m_wvalid;
  assign m_wready  = w_here ? n_wl < n_aw : s_wready;
  assign s_bready  = w_here || m_bready;
  assign m_bvalid  = w_here ? n_aw != '0 && n_wl != '0 : s_bvalid;
  assign m_bresp   = w_here ? 2'b10 : s_bresp;
  wire aw_hs = m_awvalid && m_awready;
  wire wl_hs = m_wvalid && m_wready && m_wlast;
  wire b_hs  = m_bvalid && m_bready;

  // ---------------------------------------------------------------- reads
  // lens: each outstanding read burst's ARLEN, oldest at rp; beat: the beats of the oldest the
  // master has taken. rg: the gate answers reads.
  logic          rg;
  logic [7:0]    lens [NOUT];
  logic [PW-1:0] rp, wp;
  logic [PW:0]   n_r;
  logic [7:0]    beat;
  wire r_here  = rg || dead;
  wire r_room  = n_r < (PW + 1)'(NOUT);
  wire r_last  = beat == lens[rp];
  wire ar_fwd  = !r_here && r_room && !off;
  assign s_araddr  = m_araddr;
  assign s_arlen   = m_arlen;
  assign s_arvalid = ar_fwd && m_arvalid;
  assign m_arready = r_here ? r_room && (off || dead) : ar_fwd && s_arready;
  assign s_rready  = r_here || m_rready;
  assign m_rvalid  = r_here ? n_r != '0 : s_rvalid;
  assign m_rlast   = r_here ? r_last : s_rlast;
  assign m_rresp   = r_here ? 2'b10 : s_rresp;
  assign m_rdata   = r_here ? {(DW / 32){RDATA_OFF}} : s_rdata;
  wire ar_hs = m_arvalid && m_arready;
  wire r_hs  = m_rvalid && m_rready;
  wire r_end = r_hs && r_last;              // the oldest burst's last beat (counted here)

  always_ff @(posedge clk) begin
    if (ar_hs) lens[wp] <= m_arlen;
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      wg <= 1'b0; n_aw <= '0; n_wl <= '0; w_mid <= 1'b0;
      rg <= 1'b0; rp <= '0; wp <= '0; n_r <= '0; beat <= '0;
    end else begin
      n_aw <= n_aw + CW'(aw_hs) - CW'(b_hs);
      n_wl <= n_wl + CW'(wl_hs) - CW'(b_hs);
      if (m_wvalid && m_wready) w_mid <= !m_wlast;
      // (not in a cycle in which a burst's first beat goes to the slave: its AW must follow it)
      if (dead || (off && w_clean && !(s_wvalid && s_wready))) wg <= 1'b1;
      else if (!off && w_clean) wg <= 1'b0;
      if (ar_hs) wp <= wp + 1'b1;
      if (r_end) rp <= rp + 1'b1;
      n_r <= n_r + (PW + 1)'(ar_hs) - (PW + 1)'(r_end);
      if (r_end) beat <= '0;
      else if (r_hs) beat <= beat + 1'b1;
      if (dead || (off && n_r == '0)) rg <= 1'b1;
      else if (!off && n_r == '0) rg <= 1'b0;
    end
  end
endmodule
