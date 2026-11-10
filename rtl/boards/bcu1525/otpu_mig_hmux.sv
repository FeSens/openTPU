// One DDR4 DIMM's AXI4 port on the BCU1525 at WG 8 (the block design's S_AXI_P<d>: an AXI Clock
// Converter into the DIMM's MIG, bd_wg.tcl) shared by two requesters at aclk: the DIMM's pair
// (otpu_mig_pair m_*: IDs {s, 0...}, bit IDW-1 its bridge, bit HB always 0) and the host's DMA
// (the block design's M_AXI_H<d>: XDMA's, through a SmartConnect out of xdma_aclk; no IDs, here
// ID 2^HB: HB IDW-2 for a pair of two channels, {0, 1, 0...}; 0 for one of four, whose bridges
// take the top two bits). The accelerator's stream goes through no SmartConnect: one takes L + 1 cycles
// for an L-beat burst's R (8 beats: 88.9% of the port, the card's ~86% stream ceiling; xsim,
// 2026-10-29: 32 outstanding, no narrow bursts, 2 threads the same; the AXI Interconnect 0.616),
// the clock converter a beat a cycle. The host's path keeps its SmartConnect (XDMA's width, the
// crossing from xdma_aclk), whose bubble costs the host's DMA nothing that matters.
//
// AR and AW alternate between the requesters that ask (one burst a turn, from a register, as
// otpu_mig_pair and otpu_axi3_mux2), addresses and lengths as given (AW bits: the top remaps the
// host's into the DIMM's map). W follows AW's order (AXI4 has no WID): each granted AW queues its
// requester in u_wo, and W passes the head's beats until its WLAST (the host's W may come before
// its AW: it waits). B and R go back by their ID's bit HB, the pair's with its ID (bit IDW-1
// its bridge), the host's without. W and R pass a register slice each (otpu_rs, the ring: valid
// and ready from flip-flops, the entries' enables from the ring's state alone), so the 512-bit
// routes from the converter's R to the pair's R FIFOs and from the pair's W to the converter go
// flop to flop (at 300 MHz they were aclk's critical paths), and neither the converter's WREADY
// nor the pair's RREADY reaches a data enable. The host holding its R back holds the pair's too
// (one R channel; the host's SmartConnect takes the beats into its FIFOs).
module otpu_mig_hmux #(
  parameter int IDW = 4,                   // the DIMM port's IDs
  parameter int AW  = 33,                  // the DIMM's byte address bits (the MIG's AXI side)
  parameter int HB  = IDW - 2,             // the host's ID bit (its ID 2^HB; the pair's 0 there)
  parameter int RCP = 0,                   // the W and R rings' read selects in copies (otpu_rs RCP)
  parameter int WCP = 0                    // and their entries' enables (otpu_rs WCP)
) (
  input  logic              clk,           // aclk
  input  logic              rst,
  // the pair (otpu_mig_pair m_*)
  input  logic              p_awvalid,
  output logic              p_awready,
  input  logic [IDW-1:0]    p_awid,
  input  logic [AW-1:0]     p_awaddr,
  input  logic [7:0]        p_awlen,
  input  logic              p_wvalid,
  output logic              p_wready,
  input  logic [511:0]      p_wdata,
  input  logic [63:0]       p_wstrb,
  input  logic              p_wlast,
  output logic              p_bvalid,
  input  logic              p_bready,
  output logic [IDW-1:0]    p_bid,
  output logic [1:0]        p_bresp,
  input  logic              p_arvalid,
  output logic              p_arready,
  input  logic [IDW-1:0]    p_arid,
  input  logic [AW-1:0]     p_araddr,
  input  logic [7:0]        p_arlen,
  output logic              p_rvalid,
  input  logic              p_rready,
  output logic [IDW-1:0]    p_rid,
  output logic [511:0]      p_rdata,
  output logic [1:0]        p_rresp,
  output logic              p_rlast,
  // the host's DMA (M_AXI_H<d>: 512-bit INCR bursts of whole beats, the top ties AxSIZE)
  input  logic              h_awvalid,
  output logic              h_awready,
  input  logic [AW-1:0]     h_awaddr,
  input  logic [7:0]        h_awlen,
  input  logic              h_wvalid,
  output logic              h_wready,
  input  logic [511:0]      h_wdata,
  input  logic [63:0]       h_wstrb,
  input  logic              h_wlast,
  output logic              h_bvalid,
  input  logic              h_bready,
  output logic [1:0]        h_bresp,
  input  logic              h_arvalid,
  output logic              h_arready,
  input  logic [AW-1:0]     h_araddr,
  input  logic [7:0]        h_arlen,
  output logic              h_rvalid,
  input  logic              h_rready,
  output logic [511:0]      h_rdata,
  output logic [1:0]        h_rresp,
  output logic              h_rlast,
  // the DIMM's port (S_AXI_P<d>)
  output logic              m_awvalid,
  input  logic              m_awready,
  output logic [IDW-1:0]    m_awid,
  output logic [AW-1:0]     m_awaddr,
  output logic [7:0]        m_awlen,
  output logic              m_wvalid,
  input  logic              m_wready,
  output logic [511:0]      m_wdata,
  output logic [63:0]       m_wstrb,
  output logic              m_wlast,
  input  logic              m_bvalid,
  output logic              m_bready,
  input  logic [IDW-1:0]    m_bid,
  input  logic [1:0]        m_bresp,
  output logic              m_arvalid,
  input  logic              m_arready,
  output logic [IDW-1:0]    m_arid,
  output logic [AW-1:0]     m_araddr,
  output logic [7:0]        m_arlen,
  input  logic              m_rvalid,
  output logic              m_rready,
  input  logic [IDW-1:0]    m_rid,
  input  logic [511:0]      m_rdata,
  input  logic [1:0]        m_rresp,
  input  logic              m_rlast
);
  initial if (IDW < 2 || HB < 0 || HB > IDW - 2)
    $fatal(1, "otpu_mig_hmux: IDW %0d, HB %0d (IDW 2 or more, HB 0 .. IDW - 2)", IDW, HB);
  localparam logic [IDW-1:0] HID = IDW'(1) << HB;          // the host's ID

  // ---- AR: a register, loaded from the requester whose turn it is (or the other, if only it asks)
  logic ar_last, ar_h;
  assign ar_h = h_arvalid && (!p_arvalid || !ar_last);
  assign p_arready = (!m_arvalid || m_arready) && p_arvalid && !ar_h;
  assign h_arready = (!m_arvalid || m_arready) && ar_h;
  always_ff @(posedge clk)
    if (rst) begin m_arvalid <= 1'b0; ar_last <= 1'b0; end
    else if (!m_arvalid || m_arready) begin
      m_arvalid <= p_arvalid || h_arvalid;
      if (p_arvalid || h_arvalid) ar_last <= ar_h;
    end
  always_ff @(posedge clk)                 // (the payload: no reset)
    if ((!m_arvalid || m_arready) && (p_arvalid || h_arvalid)) begin
      m_arid   <= ar_h ? HID : p_arid;
      m_araddr <= ar_h ? h_araddr : p_araddr;
      m_arlen  <= ar_h ? h_arlen : p_arlen;
    end

  // ---- AW: the same, each grant queued for W (u_wo); none while u_wo is full
  logic aw_last, aw_h, aw_room, aw_go, wo_rv, wo_wr, wo_h;
  assign aw_h    = h_awvalid && (!p_awvalid || !aw_last);
  assign aw_room = (!m_awvalid || m_awready) && wo_wr;
  assign aw_go   = aw_room && (p_awvalid || h_awvalid);
  assign p_awready = aw_room && p_awvalid && !aw_h;
  assign h_awready = aw_room && aw_h;
  always_ff @(posedge clk)
    if (rst) begin m_awvalid <= 1'b0; aw_last <= 1'b0; end
    else if (!m_awvalid || m_awready) begin
      m_awvalid <= aw_go;
      if (aw_go) aw_last <= aw_h;
    end
  always_ff @(posedge clk)
    if ((!m_awvalid || m_awready) && aw_go) begin
      m_awid   <= aw_h ? HID : p_awid;
      m_awaddr <= aw_h ? h_awaddr : p_awaddr;
      m_awlen  <= aw_h ? h_awlen : p_awlen;
    end

  // ---- W: the head's beats into the register slice; the head leaves with its WLAST
  logic w_v, w_r, w_l;
  assign w_v = wo_rv && (wo_h ? h_wvalid : p_wvalid);
  assign w_l = wo_h ? h_wlast : p_wlast;
  assign p_wready = wo_rv && !wo_h && w_r;
  assign h_wready = wo_rv && wo_h && w_r;
  otpu_sfifo #(.W(1), .DEPTH(16), .RO(1'b1)) u_wo (.clk, .rst, .wvalid(aw_go), .wready(wo_wr),
    .wdata(aw_h), .rvalid(wo_rv), .rready(w_v && w_r && w_l), .rdata(wo_h));
  otpu_rs #(.W(512 + 64 + 1), .RCP(RCP), .WCP(WCP)) u_w (.clk, .rst, .s_valid(w_v), .s_ready(w_r),
    .s_data(wo_h ? {h_wlast, h_wstrb, h_wdata} : {p_wlast, p_wstrb, p_wdata}),
    .m_valid(m_wvalid), .m_ready(m_wready), .m_data({m_wlast, m_wstrb, m_wdata}));

  // ---- B: back by BID's bit HB
  logic b_h;
  assign b_h = m_bid[HB];
  assign p_bvalid = m_bvalid && !b_h;
  assign h_bvalid = m_bvalid && b_h;
  assign p_bid    = m_bid;
  assign p_bresp  = m_bresp;
  assign h_bresp  = m_bresp;
  assign m_bready = b_h ? h_bready : p_bready;

  // ---- R: through the register slice, then by RID's bit HB
  logic           r_v, r_r, r_l;
  logic [IDW-1:0] r_id;
  logic [511:0]   r_d;
  logic [1:0]     r_e;
  otpu_rs #(.W(IDW + 512 + 2 + 1), .RCP(RCP), .WCP(WCP)) u_r (.clk, .rst, .s_valid(m_rvalid),
    .s_ready(m_rready), .s_data({m_rid, m_rlast, m_rresp, m_rdata}), .m_valid(r_v), .m_ready(r_r),
    .m_data({r_id, r_l, r_e, r_d}));
  assign r_r      = r_id[HB] ? h_rready : p_rready;
  assign p_rvalid = r_v && !r_id[HB];
  assign h_rvalid = r_v && r_id[HB];
  assign p_rid    = r_id;
  assign p_rdata  = r_d;
  assign h_rdata  = r_d;
  assign p_rresp  = r_e;
  assign h_rresp  = r_e;
  assign p_rlast  = r_l;
  assign h_rlast  = r_l;

`ifndef SYNTHESIS
  always_ff @(posedge clk)
    if (!rst) begin
      if (p_arvalid && p_arid[HB])
        $error("otpu_mig_hmux: the pair's AR with ID %h (bit %0d is the host's)", p_arid, HB);
      if (p_awvalid && p_awid[HB])
        $error("otpu_mig_hmux: the pair's AW with ID %h (bit %0d is the host's)", p_awid, HB);
      if (m_rvalid && m_rid[HB] && m_rid != HID)
        $error("otpu_mig_hmux: R with ID %h (the host's is %h)", m_rid, HID);
      if (m_bvalid && m_bid[HB] && m_bid != HID)
        $error("otpu_mig_hmux: B with ID %h (the host's is %h)", m_bid, HID);
    end
`endif
endmodule
