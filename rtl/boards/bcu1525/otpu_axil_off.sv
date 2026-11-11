// AXI4-Lite in front of a slave that can be switched off (the BCU1525's tops: a DIMM's LiteDRAM
// core's CSRs, behind otpu_axil_iso, while the governor holds the core in reset and its clocks are
// stopped; otpu_bcu_gov). While `off`, every access is answered at once with SLVERR (reads
// 0xDEAD0FF0) instead of waiting for a slave that cannot answer, so the host's JTAG or PCIe master
// is never left with an access in flight. One access at a time, as otpu_axil_iso. `off` takes
// effect between accesses: one already with the slave is completed there (otpu_axil_iso answers it
// SLVERR when its s_rst rises with `off`); new ones wait for the switch. A valid the master raised
// is withdrawn from the slave when `off` rises before the slave took it: otpu_axil_iso keeps no
// state for an access until it takes it, so this is safe in front of it (not for any slave).
// ERR_RESP: the response while off. SLVERR (the default) for a JTAG master, which reports it; the
// full image over PCIe answers OKAY (reads 0xDEAD0FF0): XDMA turns an AXI-Lite error into a
// completer abort, which the host's AER handling may act on, and the host reads the governor first.
module otpu_axil_off #(
  parameter int AW = 16,
  parameter logic [1:0] ERR_RESP = 2'b10
) (
  input  logic          clk,
  input  logic          rst,
  input  logic          off,
  // the master's side
  input  logic [AW-1:0] m_awaddr,
  input  logic          m_awvalid,
  output logic          m_awready,
  input  logic [31:0]   m_wdata,
  input  logic [3:0]    m_wstrb,
  input  logic          m_wvalid,
  output logic          m_wready,
  output logic [1:0]    m_bresp,
  output logic          m_bvalid,
  input  logic          m_bready,
  input  logic [AW-1:0] m_araddr,
  input  logic          m_arvalid,
  output logic          m_arready,
  output logic [31:0]   m_rdata,
  output logic [1:0]    m_rresp,
  output logic          m_rvalid,
  input  logic          m_rready,
  // the slave's side (otpu_axil_iso)
  output logic [AW-1:0] s_awaddr,
  output logic          s_awvalid,
  input  logic          s_awready,
  output logic [31:0]   s_wdata,
  output logic [3:0]    s_wstrb,
  output logic          s_wvalid,
  input  logic          s_wready,
  input  logic [1:0]    s_bresp,
  input  logic          s_bvalid,
  output logic          s_bready,
  output logic [AW-1:0] s_araddr,
  output logic          s_arvalid,
  input  logic          s_arready,
  input  logic [31:0]   s_rdata,
  input  logic [1:0]    s_rresp,
  input  logic          s_rvalid,
  output logic          s_rready
);
  // err: answering here; out: an access with the slave (taken by it, its response not yet taken
  // by the master); e_b / e_r: an error response held for the master
  logic err, out, e_b, e_r;
  wire pass = !err && !off;
  wire e_aw = err && off && !e_b && !e_r && m_awvalid && m_wvalid;
  wire e_ar = err && off && !e_b && !e_r && m_arvalid && !(m_awvalid && m_wvalid);

  assign s_awaddr  = m_awaddr;
  assign s_wdata   = m_wdata;
  assign s_wstrb   = m_wstrb;
  assign s_araddr  = m_araddr;
  assign s_awvalid = pass && m_awvalid;
  assign s_wvalid  = pass && m_wvalid;
  assign s_arvalid = pass && m_arvalid;
  assign m_awready = err ? e_aw : pass && s_awready;
  assign m_wready  = err ? e_aw : pass && s_wready;
  assign m_arready = err ? e_ar : pass && s_arready;
  assign s_bready  = !err && m_bready;
  assign s_rready  = !err && m_rready;
  assign m_bvalid  = err ? e_b : s_bvalid;
  assign m_bresp   = err ? ERR_RESP : s_bresp;
  assign m_rvalid  = err ? e_r : s_rvalid;
  assign m_rresp   = err ? ERR_RESP : s_rresp;
  assign m_rdata   = err ? 32'hDEAD_0FF0 : s_rdata;

  always_ff @(posedge clk) begin
    if (rst) begin
      err <= 1'b0; out <= 1'b0; e_b <= 1'b0; e_r <= 1'b0;
    end else begin
      if ((s_awvalid && s_awready) || (s_arvalid && s_arready)) out <= 1'b1;
      if (!err && ((m_bvalid && m_bready) || (m_rvalid && m_rready))) out <= 1'b0;
      if (e_aw) e_b <= 1'b1;
      if (e_ar) e_r <= 1'b1;
      if (err && m_bvalid && m_bready) e_b <= 1'b0;
      if (err && m_rvalid && m_rready) e_r <= 1'b0;
      // the switch, between accesses: to answering once the slave has none, back once no error
      // response is held
      if (!err && off && !out) err <= 1'b1;
      if (err && !off && !e_b && !e_r) err <= 1'b0;
    end
  end
endmodule
