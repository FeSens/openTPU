// AXI4 1-to-2 split by address bit 31 (XDMA's M_AXI onto the two channels' bridges,
// otpu_mem_ch; channel 0 at 0x0000_0000, channel 1 at 0x8000_0000, as the address map of
// opentpu/host/board.py). One clock. W beats follow their AW's channel; B and R responses come
// back in the order of their AW / AR, whatever the IDs (legal, and XDMA's own order), so a
// response from the other channel waits for the older ones. Up to OD transactions of each kind
// in flight. Valids to the channels depend on no ready.
module otpu_axi_split2 #(
  parameter int IDW = 4,
  parameter int DW  = 128,
  parameter int OD  = 16
) (
  input  logic                  clk,
  input  logic                  rst,
  // slave (XDMA)
  input  logic                  s_awvalid,
  output logic                  s_awready,
  input  logic [IDW-1:0]        s_awid,
  input  logic [31:0]           s_awaddr,
  input  logic [7:0]            s_awlen,
  input  logic                  s_wvalid,
  output logic                  s_wready,
  input  logic [DW-1:0]         s_wdata,
  input  logic [DW/8-1:0]       s_wstrb,
  input  logic                  s_wlast,
  output logic                  s_bvalid,
  input  logic                  s_bready,
  output logic [IDW-1:0]        s_bid,
  output logic [1:0]            s_bresp,
  input  logic                  s_arvalid,
  output logic                  s_arready,
  input  logic [IDW-1:0]        s_arid,
  input  logic [31:0]           s_araddr,
  input  logic [7:0]            s_arlen,
  output logic                  s_rvalid,
  input  logic                  s_rready,
  output logic [IDW-1:0]        s_rid,
  output logic [DW-1:0]         s_rdata,
  output logic [1:0]            s_rresp,
  output logic                  s_rlast,
  // masters, one per channel
  output logic [1:0]            m_awvalid,
  input  logic [1:0]            m_awready,
  output logic [1:0]            m_wvalid,
  input  logic [1:0]            m_wready,
  input  logic [1:0]            m_bvalid,
  output logic [1:0]            m_bready,
  input  logic [1:0][IDW-1:0]   m_bid,
  input  logic [1:0][1:0]       m_bresp,
  output logic [1:0]            m_arvalid,
  input  logic [1:0]            m_arready,
  input  logic [1:0]            m_rvalid,
  output logic [1:0]            m_rready,
  input  logic [1:0][IDW-1:0]   m_rid,
  input  logic [1:0][DW-1:0]    m_rdata,
  input  logic [1:0][1:0]       m_rresp,
  input  logic [1:0]            m_rlast
);
  // the address, ID, length and write data go to both channels as they are (only the valids
  // select one)
  logic ow_wr, ow_rv, ow_rd;              // W route (AW order)
  logic ob_wr, ob_rv, ob_rd;              // B route (AW order)
  logic or_wr, or_rv, or_rd;              // R route (AR order)
  logic awc, arc;
  assign awc = s_awaddr[31];
  assign arc = s_araddr[31];

  otpu_sfifo #(.W(1), .DEPTH(OD)) u_ow (.clk, .rst, .wvalid(s_awvalid && s_awready), .wready(ow_wr),
    .wdata(awc), .rvalid(ow_rv), .rready(s_wvalid && s_wready && s_wlast), .rdata(ow_rd));
  otpu_sfifo #(.W(1), .DEPTH(OD)) u_ob (.clk, .rst, .wvalid(s_awvalid && s_awready), .wready(ob_wr),
    .wdata(awc), .rvalid(ob_rv), .rready(s_bvalid && s_bready), .rdata(ob_rd));
  otpu_sfifo #(.W(1), .DEPTH(OD)) u_or (.clk, .rst, .wvalid(s_arvalid && s_arready), .wready(or_wr),
    .wdata(arc), .rvalid(or_rv), .rready(s_rvalid && s_rready && s_rlast), .rdata(or_rd));

  assign m_awvalid = {s_awvalid && ow_wr && ob_wr && awc, s_awvalid && ow_wr && ob_wr && !awc};
  assign s_awready = ow_wr && ob_wr && m_awready[awc];
  assign m_wvalid  = {s_wvalid && ow_rv && ow_rd, s_wvalid && ow_rv && !ow_rd};
  assign s_wready  = ow_rv && m_wready[ow_rd];
  assign s_bvalid  = ob_rv && m_bvalid[ob_rd];
  assign s_bid     = m_bid[ob_rd];
  assign s_bresp   = m_bresp[ob_rd];
  assign m_bready  = {s_bready && ob_rv && ob_rd, s_bready && ob_rv && !ob_rd};
  assign m_arvalid = {s_arvalid && or_wr && arc, s_arvalid && or_wr && !arc};
  assign s_arready = or_wr && m_arready[arc];
  assign s_rvalid  = or_rv && m_rvalid[or_rd];
  assign s_rid     = m_rid[or_rd];
  assign s_rdata   = m_rdata[or_rd];
  assign s_rresp   = m_rresp[or_rd];
  assign s_rlast   = m_rlast[or_rd];
  assign m_rready  = {s_rready && or_rv && or_rd, s_rready && or_rv && !or_rd};
endmodule

// One AXI4 channel's register slice: the payload, its valid and its ready all from flip-flops
// (an output register and a skid entry), a beat per cycle at full rate, one cycle of latency. In
// reset it takes nothing (s_ready low) and drops what it holds.
module otpu_skid #(
  parameter int W = 8
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         s_valid,
  output logic         s_ready,
  input  logic [W-1:0] s_data,
  output logic         m_valid,
  input  logic         m_ready,
  output logic [W-1:0] m_data
);
  logic         sk_v;
  logic [W-1:0] sk_d;
  logic         adv, take;
  assign adv  = m_ready || !m_valid;      // the output register takes a beat
  assign take = s_valid && s_ready;
  always_ff @(posedge clk) begin
    if (rst) begin m_valid <= 1'b0; sk_v <= 1'b0; s_ready <= 1'b0; end
    else if (adv) begin m_valid <= sk_v || take; sk_v <= 1'b0; s_ready <= 1'b1; end
    else if (take) begin sk_v <= 1'b1; s_ready <= 1'b0; end
    else s_ready <= !sk_v;
  end
  always_ff @(posedge clk) begin
    if (adv) m_data <= sk_v ? sk_d : s_data;
    if (!adv && take) sk_d <= s_data;
  end
endmodule

// An AXI4 register slice: otpu_skid on each of the five channels (otpu_dma_split). REG = 0: wires.
module otpu_axi_slice #(
  parameter bit REG = 1'b1,
  parameter int IDW = 4,
  parameter int DW  = 128
) (
  input  logic                  clk,
  input  logic                  rst,
  // slave
  input  logic                  s_awvalid,
  output logic                  s_awready,
  input  logic [IDW-1:0]        s_awid,
  input  logic [31:0]           s_awaddr,
  input  logic [7:0]            s_awlen,
  input  logic                  s_wvalid,
  output logic                  s_wready,
  input  logic [DW-1:0]         s_wdata,
  input  logic [DW/8-1:0]       s_wstrb,
  input  logic                  s_wlast,
  output logic                  s_bvalid,
  input  logic                  s_bready,
  output logic [IDW-1:0]        s_bid,
  output logic [1:0]            s_bresp,
  input  logic                  s_arvalid,
  output logic                  s_arready,
  input  logic [IDW-1:0]        s_arid,
  input  logic [31:0]           s_araddr,
  input  logic [7:0]            s_arlen,
  output logic                  s_rvalid,
  input  logic                  s_rready,
  output logic [IDW-1:0]        s_rid,
  output logic [DW-1:0]         s_rdata,
  output logic [1:0]            s_rresp,
  output logic                  s_rlast,
  // master
  output logic                  m_awvalid,
  input  logic                  m_awready,
  output logic [IDW-1:0]        m_awid,
  output logic [31:0]           m_awaddr,
  output logic [7:0]            m_awlen,
  output logic                  m_wvalid,
  input  logic                  m_wready,
  output logic [DW-1:0]         m_wdata,
  output logic [DW/8-1:0]       m_wstrb,
  output logic                  m_wlast,
  input  logic                  m_bvalid,
  output logic                  m_bready,
  input  logic [IDW-1:0]        m_bid,
  input  logic [1:0]            m_bresp,
  output logic                  m_arvalid,
  input  logic                  m_arready,
  output logic [IDW-1:0]        m_arid,
  output logic [31:0]           m_araddr,
  output logic [7:0]            m_arlen,
  input  logic                  m_rvalid,
  output logic                  m_rready,
  input  logic [IDW-1:0]        m_rid,
  input  logic [DW-1:0]         m_rdata,
  input  logic [1:0]            m_rresp,
  input  logic                  m_rlast
);
  if (REG) begin : g_reg
    otpu_skid #(.W(IDW + 40)) u_aw (.clk, .rst, .s_valid(s_awvalid), .s_ready(s_awready),
      .s_data({s_awid, s_awaddr, s_awlen}), .m_valid(m_awvalid), .m_ready(m_awready),
      .m_data({m_awid, m_awaddr, m_awlen}));
    otpu_skid #(.W(DW + DW / 8 + 1)) u_w (.clk, .rst, .s_valid(s_wvalid), .s_ready(s_wready),
      .s_data({s_wdata, s_wstrb, s_wlast}), .m_valid(m_wvalid), .m_ready(m_wready),
      .m_data({m_wdata, m_wstrb, m_wlast}));
    otpu_skid #(.W(IDW + 2)) u_b (.clk, .rst, .s_valid(m_bvalid), .s_ready(m_bready),
      .s_data({m_bid, m_bresp}), .m_valid(s_bvalid), .m_ready(s_bready),
      .m_data({s_bid, s_bresp}));
    otpu_skid #(.W(IDW + 40)) u_ar (.clk, .rst, .s_valid(s_arvalid), .s_ready(s_arready),
      .s_data({s_arid, s_araddr, s_arlen}), .m_valid(m_arvalid), .m_ready(m_arready),
      .m_data({m_arid, m_araddr, m_arlen}));
    otpu_skid #(.W(IDW + DW + 3)) u_r (.clk, .rst, .s_valid(m_rvalid), .s_ready(m_rready),
      .s_data({m_rid, m_rdata, m_rresp, m_rlast}), .m_valid(s_rvalid), .m_ready(s_rready),
      .m_data({s_rid, s_rdata, s_rresp, s_rlast}));
  end else begin : g_wire
    assign {m_awvalid, m_awid, m_awaddr, m_awlen} = {s_awvalid, s_awid, s_awaddr, s_awlen};
    assign s_awready = m_awready;
    assign {m_wvalid, m_wdata, m_wstrb, m_wlast} = {s_wvalid, s_wdata, s_wstrb, s_wlast};
    assign s_wready = m_wready;
    assign {s_bvalid, s_bid, s_bresp} = {m_bvalid, m_bid, m_bresp};
    assign m_bready = s_bready;
    assign {m_arvalid, m_arid, m_araddr, m_arlen} = {s_arvalid, s_arid, s_araddr, s_arlen};
    assign s_arready = m_arready;
    assign {s_rvalid, s_rid, s_rdata, s_rresp, s_rlast} =
           {m_rvalid, m_rid, m_rdata, m_rresp, m_rlast};
    assign m_rready = s_rready;
  end
endmodule

// XDMA's DMA master onto the two channels' bridges (otpu_native_sys): otpu_axi_split2 and, with
// REG (PCIe Gen2: xdma_aclk at 250 MHz), a register slice (otpu_axi_slice) on each side of it, at
// XDMA's and at each bridge's, so no path between XDMA and a bridge, which sits by its DDR3 bank,
// is combinational at either end (two cycles of latency each way). REG = 0: the split alone.
module otpu_dma_split #(
  parameter bit REG = 1'b0,
  parameter int IDW = 4,
  parameter int DW  = 128
) (
  input  logic                  clk,
  input  logic                  rst,
  // slave (XDMA)
  input  logic                  s_awvalid,
  output logic                  s_awready,
  input  logic [IDW-1:0]        s_awid,
  input  logic [31:0]           s_awaddr,
  input  logic [7:0]            s_awlen,
  input  logic                  s_wvalid,
  output logic                  s_wready,
  input  logic [DW-1:0]         s_wdata,
  input  logic [DW/8-1:0]       s_wstrb,
  input  logic                  s_wlast,
  output logic                  s_bvalid,
  input  logic                  s_bready,
  output logic [IDW-1:0]        s_bid,
  output logic [1:0]            s_bresp,
  input  logic                  s_arvalid,
  output logic                  s_arready,
  input  logic [IDW-1:0]        s_arid,
  input  logic [31:0]           s_araddr,
  input  logic [7:0]            s_arlen,
  output logic                  s_rvalid,
  input  logic                  s_rready,
  output logic [IDW-1:0]        s_rid,
  output logic [DW-1:0]         s_rdata,
  output logic [1:0]            s_rresp,
  output logic                  s_rlast,
  // masters, one per channel (the bridges)
  output logic [1:0]            m_awvalid,
  input  logic [1:0]            m_awready,
  output logic [1:0][IDW-1:0]   m_awid,
  output logic [1:0][31:0]      m_awaddr,
  output logic [1:0][7:0]       m_awlen,
  output logic [1:0]            m_wvalid,
  input  logic [1:0]            m_wready,
  output logic [1:0][DW-1:0]    m_wdata,
  output logic [1:0][DW/8-1:0]  m_wstrb,
  output logic [1:0]            m_wlast,
  input  logic [1:0]            m_bvalid,
  output logic [1:0]            m_bready,
  input  logic [1:0][IDW-1:0]   m_bid,
  input  logic [1:0][1:0]       m_bresp,
  output logic [1:0]            m_arvalid,
  input  logic [1:0]            m_arready,
  output logic [1:0][IDW-1:0]   m_arid,
  output logic [1:0][31:0]      m_araddr,
  output logic [1:0][7:0]       m_arlen,
  input  logic [1:0]            m_rvalid,
  output logic [1:0]            m_rready,
  input  logic [1:0][IDW-1:0]   m_rid,
  input  logic [1:0][DW-1:0]    m_rdata,
  input  logic [1:0][1:0]       m_rresp,
  input  logic [1:0]            m_rlast
);
  // XDMA's side of the split (i_*) and its masters (j_*; the address, ID, length and write data
  // go to both from i_*)
  logic             i_awvalid, i_awready, i_wvalid, i_wready, i_wlast, i_bvalid, i_bready;
  logic             i_arvalid, i_arready, i_rvalid, i_rready, i_rlast;
  logic [IDW-1:0]   i_awid, i_arid, i_bid, i_rid;
  logic [31:0]      i_awaddr, i_araddr;
  logic [7:0]       i_awlen, i_arlen;
  logic [DW-1:0]    i_wdata, i_rdata;
  logic [DW/8-1:0]  i_wstrb;
  logic [1:0]       i_bresp, i_rresp;
  logic [1:0]           j_awvalid, j_awready, j_wvalid, j_wready, j_bvalid, j_bready;
  logic [1:0]           j_arvalid, j_arready, j_rvalid, j_rready, j_rlast;
  logic [1:0][IDW-1:0]  j_bid, j_rid;
  logic [1:0][1:0]      j_bresp, j_rresp;
  logic [1:0][DW-1:0]   j_rdata;

  otpu_axi_slice #(.REG(REG), .IDW(IDW), .DW(DW)) u_xs (.clk, .rst,
    .s_awvalid, .s_awready, .s_awid, .s_awaddr, .s_awlen, .s_wvalid, .s_wready, .s_wdata,
    .s_wstrb, .s_wlast, .s_bvalid, .s_bready, .s_bid, .s_bresp, .s_arvalid, .s_arready, .s_arid,
    .s_araddr, .s_arlen, .s_rvalid, .s_rready, .s_rid, .s_rdata, .s_rresp, .s_rlast,
    .m_awvalid(i_awvalid), .m_awready(i_awready), .m_awid(i_awid), .m_awaddr(i_awaddr),
    .m_awlen(i_awlen), .m_wvalid(i_wvalid), .m_wready(i_wready), .m_wdata(i_wdata),
    .m_wstrb(i_wstrb), .m_wlast(i_wlast), .m_bvalid(i_bvalid), .m_bready(i_bready),
    .m_bid(i_bid), .m_bresp(i_bresp), .m_arvalid(i_arvalid), .m_arready(i_arready),
    .m_arid(i_arid), .m_araddr(i_araddr), .m_arlen(i_arlen), .m_rvalid(i_rvalid),
    .m_rready(i_rready), .m_rid(i_rid), .m_rdata(i_rdata), .m_rresp(i_rresp), .m_rlast(i_rlast));

  otpu_axi_split2 #(.IDW(IDW), .DW(DW)) u_split (.clk, .rst,
    .s_awvalid(i_awvalid), .s_awready(i_awready), .s_awid(i_awid), .s_awaddr(i_awaddr),
    .s_awlen(i_awlen), .s_wvalid(i_wvalid), .s_wready(i_wready), .s_wdata(i_wdata),
    .s_wstrb(i_wstrb), .s_wlast(i_wlast), .s_bvalid(i_bvalid), .s_bready(i_bready),
    .s_bid(i_bid), .s_bresp(i_bresp), .s_arvalid(i_arvalid), .s_arready(i_arready),
    .s_arid(i_arid), .s_araddr(i_araddr), .s_arlen(i_arlen), .s_rvalid(i_rvalid),
    .s_rready(i_rready), .s_rid(i_rid), .s_rdata(i_rdata), .s_rresp(i_rresp), .s_rlast(i_rlast),
    .m_awvalid(j_awvalid), .m_awready(j_awready), .m_wvalid(j_wvalid), .m_wready(j_wready),
    .m_bvalid(j_bvalid), .m_bready(j_bready), .m_bid(j_bid), .m_bresp(j_bresp),
    .m_arvalid(j_arvalid), .m_arready(j_arready), .m_rvalid(j_rvalid), .m_rready(j_rready),
    .m_rid(j_rid), .m_rdata(j_rdata), .m_rresp(j_rresp), .m_rlast(j_rlast));

  for (genvar c = 0; c < 2; c++) begin : g_cs
    otpu_axi_slice #(.REG(REG), .IDW(IDW), .DW(DW)) u_cs (.clk, .rst,
      .s_awvalid(j_awvalid[c]), .s_awready(j_awready[c]), .s_awid(i_awid), .s_awaddr(i_awaddr),
      .s_awlen(i_awlen), .s_wvalid(j_wvalid[c]), .s_wready(j_wready[c]), .s_wdata(i_wdata),
      .s_wstrb(i_wstrb), .s_wlast(i_wlast), .s_bvalid(j_bvalid[c]), .s_bready(j_bready[c]),
      .s_bid(j_bid[c]), .s_bresp(j_bresp[c]), .s_arvalid(j_arvalid[c]),
      .s_arready(j_arready[c]), .s_arid(i_arid), .s_araddr(i_araddr), .s_arlen(i_arlen),
      .s_rvalid(j_rvalid[c]), .s_rready(j_rready[c]), .s_rid(j_rid[c]), .s_rdata(j_rdata[c]),
      .s_rresp(j_rresp[c]), .s_rlast(j_rlast[c]),
      .m_awvalid(m_awvalid[c]), .m_awready(m_awready[c]), .m_awid(m_awid[c]),
      .m_awaddr(m_awaddr[c]), .m_awlen(m_awlen[c]), .m_wvalid(m_wvalid[c]),
      .m_wready(m_wready[c]), .m_wdata(m_wdata[c]), .m_wstrb(m_wstrb[c]), .m_wlast(m_wlast[c]),
      .m_bvalid(m_bvalid[c]), .m_bready(m_bready[c]), .m_bid(m_bid[c]), .m_bresp(m_bresp[c]),
      .m_arvalid(m_arvalid[c]), .m_arready(m_arready[c]), .m_arid(m_arid[c]),
      .m_araddr(m_araddr[c]), .m_arlen(m_arlen[c]), .m_rvalid(m_rvalid[c]),
      .m_rready(m_rready[c]), .m_rid(m_rid[c]), .m_rdata(m_rdata[c]), .m_rresp(m_rresp[c]),
      .m_rlast(m_rlast[c]));
  end
endmodule
