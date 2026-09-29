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
