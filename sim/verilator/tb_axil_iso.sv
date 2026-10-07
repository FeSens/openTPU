// Unit test of otpu_axil_iso (rtl/boards/ypcb-00338/otpu_ctrl.sv; tests/test_board.py): a master
// with a write and a read channel of its own (AW and W given apart, each held until taken, B / R
// taken at random) and a register-file slave (AW and W taken apart, answers after a random
// latency, a read's data tagged with its address), each reset at random apart from the other.
// Checked: the master never gets a response it did not ask for, nor a read's data for another
// address, and every access it gave gets one within TO cycles (unless it is reset first: a slave
// reset is answered SLVERR); the slave never holds half a write (AW without its W, or W without
// its AW) for TO cycles, nor a response nobody takes. BYPASS = 1 connects master and slave without
// the shim (as the CSR port was): the same traffic fails.
// Plusargs: +seed=, +cycles=, +mrst=, +slvrst= (the chance per 10000 cycles that a master / slave
// reset starts; a master reset lasts 1 to 8 cycles, a slave reset 1 to 20). Prints the counts
// and PASS, or the first errors and FAIL.
module tb_axil_iso;
  parameter int BYPASS = 0;
  localparam int AW = 12, TO = 3000;

  logic clk = 1'b0;
  always #5 clk = !clk;
  logic m_rst = 1'b1, s_rst = 1'b1;

  logic [AW-1:0] m_awaddr, m_araddr, s_awaddr, s_araddr;
  logic          m_awvalid, m_awready, m_wvalid, m_wready, m_bvalid, m_bready;
  logic          m_arvalid, m_arready, m_rvalid, m_rready;
  logic [31:0]   m_wdata, m_rdata, s_wdata, s_rdata;
  logic [3:0]    m_wstrb, s_wstrb;
  logic [1:0]    m_bresp, m_rresp, s_bresp, s_rresp;
  logic          s_awvalid, s_awready, s_wvalid, s_wready, s_bvalid, s_bready;
  logic          s_arvalid, s_arready, s_rvalid, s_rready;

  if (BYPASS) begin : g_bypass
    assign s_awaddr = m_awaddr; assign s_awvalid = m_awvalid; assign m_awready = s_awready;
    assign s_wdata = m_wdata; assign s_wstrb = m_wstrb; assign s_wvalid = m_wvalid;
    assign m_wready = s_wready;
    assign m_bresp = s_bresp; assign m_bvalid = s_bvalid; assign s_bready = m_bready;
    assign s_araddr = m_araddr; assign s_arvalid = m_arvalid; assign m_arready = s_arready;
    assign m_rdata = s_rdata; assign m_rresp = s_rresp; assign m_rvalid = s_rvalid;
    assign s_rready = m_rready;
  end else begin : g_iso
    otpu_axil_iso #(.AW(AW)) dut (.clk, .m_rst, .s_rst,
      .m_awaddr, .m_awvalid, .m_awready, .m_wdata, .m_wstrb, .m_wvalid, .m_wready, .m_bresp,
      .m_bvalid, .m_bready, .m_araddr, .m_arvalid, .m_arready, .m_rdata, .m_rresp, .m_rvalid,
      .m_rready,
      .s_awaddr, .s_awvalid, .s_awready, .s_wdata, .s_wstrb, .s_wvalid, .s_wready, .s_bresp,
      .s_bvalid, .s_bready, .s_araddr, .s_arvalid, .s_arready, .s_rdata, .s_rresp, .s_rvalid,
      .s_rready);
  end

  int m_err = 0, s_err = 0, cycles = 200000, mrst_p = 20, srst_p = 5;
  int n_wr = 0, n_rd = 0, n_slverr = 0, n_mrst = 0, n_srst = 0, n_sw = 0, n_sr = 0;
  task automatic mfail(input string s);
    m_err++;
    if (m_err <= 10) $display("ERROR at %0t: %s", $time, s);
  endtask
  task automatic sfail(input string s);
    s_err++;
    if (s_err <= 10) $display("ERROR at %0t: %s", $time, s);
  endtask

  // ---------------------------------------------------------------- resets
  int mrst_n = 0, srst_n = 0;
  always_ff @(posedge clk) begin
    if (mrst_n > 0) mrst_n <= mrst_n - 1;
    else if (int'($urandom % 10000) < mrst_p) begin mrst_n <= 1 + $urandom % 8; n_mrst++; end
    if (srst_n > 0) srst_n <= srst_n - 1;
    else if (int'($urandom % 10000) < srst_p) begin srst_n <= 1 + $urandom % 20; n_srst++; end
  end
  logic started = 1'b0;
  always_ff @(posedge clk) begin
    m_rst <= !started || mrst_n > 0;
    s_rst <= !started || srst_n > 0;
  end

  // ---------------------------------------------------------------- master
  // per channel: pend (an access given and not yet answered), its age, whether a slave reset came
  // while it was out (SLVERR then allowed); the address stays in m_awaddr / m_araddr
  logic w_pend = 1'b0, r_pend = 1'b0, w_sr = 1'b0, r_sr = 1'b0, w_gave = 1'b0;
  int   w_age = 0, r_age = 0;
  always_ff @(posedge clk) begin
    m_bready <= ($urandom % 100) < 70;
    m_rready <= ($urandom % 100) < 70;
    if (m_rst) begin
      m_awvalid <= 1'b0; m_wvalid <= 1'b0; m_arvalid <= 1'b0;
      w_pend <= 1'b0; r_pend <= 1'b0; m_bready <= 1'b0; m_rready <= 1'b0;
    end else begin
      // a write: AW, then W (or both at once); each held until taken
      if (m_awvalid && m_awready) m_awvalid <= 1'b0;
      if (m_wvalid && m_wready) m_wvalid <= 1'b0;
      if (!w_pend && ($urandom % 100) < 10) begin
        w_pend <= 1'b1; w_sr <= 1'b0; w_age <= 0;
        m_awaddr <= AW'({$urandom % 64, 2'b00}); m_awvalid <= 1'b1;
        m_wdata <= $urandom; m_wstrb <= 4'hf; m_wvalid <= 1'b0; w_gave <= 1'b0;
      end else if (w_pend) begin
        w_age <= w_age + 1;
        if (!w_gave && ($urandom % 4) == 0) begin m_wvalid <= 1'b1; w_gave <= 1'b1; end
      end
      if (!r_pend && ($urandom % 100) < 10) begin
        r_pend <= 1'b1; r_sr <= 1'b0; r_age <= 0;
        m_araddr <= AW'({$urandom % 64, 2'b00}); m_arvalid <= 1'b1;
      end else if (r_pend) r_age <= r_age + 1;
      if (m_arvalid && m_arready) m_arvalid <= 1'b0;
      if (s_rst) begin w_sr <= w_pend; r_sr <= r_pend; end
      // responses
      if (m_bvalid && m_bready) begin
        if (!w_pend || m_awvalid || !w_gave || m_wvalid) mfail("a B the master did not ask for");
        else if (m_bresp != 2'b00 && !w_sr) mfail("SLVERR without a slave reset");
        if (m_bresp != 2'b00) n_slverr++;
        w_pend <= 1'b0; n_wr++;
      end
      if (m_rvalid && m_rready) begin
        if (!r_pend || m_arvalid) mfail("an R the master did not ask for");
        else if (m_rresp != 2'b00) begin
          if (!r_sr) mfail("SLVERR without a slave reset");
          n_slverr++;
        end else if (m_rdata[31:16] != 16'(m_araddr))
          mfail($sformatf("R for address %h, read %h", m_rdata[31:16], m_araddr));
        r_pend <= 1'b0; n_rd++;
      end
      if (w_pend && w_age == TO) mfail("a write not answered");
      if (r_pend && r_age == TO) mfail("a read not answered");
    end
  end

  // ---------------------------------------------------------------- slave
  logic [31:0]   regs [64];
  logic          aw_have = 1'b0, w_have = 1'b0, ar_have = 1'b0;
  logic [AW-1:0] sa_w, sa_r;
  logic [31:0]   sd_w;
  int            lat = 0, half = 0, stuck = 0;
  always_ff @(posedge clk) begin
    if (s_rst) begin
      aw_have <= 1'b0; w_have <= 1'b0; ar_have <= 1'b0; s_bvalid <= 1'b0; s_rvalid <= 1'b0;
      s_awready <= 1'b0; s_wready <= 1'b0; s_arready <= 1'b0; half <= 0; stuck <= 0;
    end else begin
      s_awready <= !aw_have && !ar_have && !s_bvalid && !s_rvalid && ($urandom % 100) < 50;
      s_wready  <= !w_have && !ar_have && !s_bvalid && !s_rvalid && ($urandom % 100) < 50;
      s_arready <= !aw_have && !w_have && !ar_have && !s_bvalid && !s_rvalid && ($urandom % 100) < 50;
      if (s_awvalid && s_awready) begin aw_have <= 1'b1; sa_w <= s_awaddr; s_awready <= 1'b0; end
      if (s_wvalid && s_wready) begin w_have <= 1'b1; sd_w <= s_wdata; s_wready <= 1'b0; end
      if (s_arvalid && s_arready) begin
        ar_have <= 1'b1; sa_r <= s_araddr; s_arready <= 1'b0; s_awready <= 1'b0;
        s_wready <= 1'b0; lat <= $urandom % 30;
      end
      if (aw_have && w_have && !s_bvalid) begin
        if (lat > 0) lat <= lat - 1;
        else begin
          regs[sa_w[7:2]] <= sd_w; s_bvalid <= 1'b1; s_bresp <= 2'b00; n_sw++;
        end
      end
      if (ar_have && !s_rvalid) begin
        if (lat > 0) lat <= lat - 1;
        else begin
          s_rvalid <= 1'b1; s_rresp <= 2'b00; s_rdata <= {16'(sa_r), regs[sa_r[7:2]][15:0]};
          n_sr++;
        end
      end
      if (s_bvalid && s_bready) begin
        s_bvalid <= 1'b0; aw_have <= 1'b0; w_have <= 1'b0; lat <= $urandom % 30;
      end
      if (s_rvalid && s_rready) begin s_rvalid <= 1'b0; ar_have <= 1'b0; lat <= $urandom % 30; end
      half <= (aw_have != w_have) ? half + 1 : 0;
      stuck <= ((s_bvalid && !s_bready) || (s_rvalid && !s_rready)) ? stuck + 1 : 0;
      if (half == TO) sfail("half a write with the slave");
      if (stuck == TO) sfail("a slave response nobody takes");
    end
  end

  initial begin
    int seed;
    if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));
    void'($value$plusargs("cycles=%d", cycles));
    void'($value$plusargs("mrst=%d", mrst_p));
    void'($value$plusargs("slvrst=%d", srst_p));
    for (int i = 0; i < 64; i++) regs[i] = '0;
    repeat (4) @(posedge clk);
    started = 1'b1;
    repeat (cycles) @(posedge clk);
    $display("writes %0d reads %0d slverr %0d master resets %0d slave resets %0d slave writes %0d reads %0d",
             n_wr, n_rd, n_slverr, n_mrst, n_srst, n_sw, n_sr);
    if (m_err + s_err == 0 && n_wr > 1000 && n_rd > 1000 && n_mrst > 10) $display("PASS");
    else $display("FAIL (%0d errors)", m_err + s_err);
    $finish;
  end
endmodule
