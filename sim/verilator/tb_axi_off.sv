// Unit test of otpu_axi_off (rtl/boards/bcu1525/otpu_axi_off.sv; tests/test_bcu_gov.py): random
// AXI4 bursts (no IDs) from a master whose write data may lead its address, through the gate, to a
// slave model that answers in order with random gaps, and freezes as a stopped clock does while
// `frozen` (its state and outputs held), and is reset before it runs again. The scenario: traffic;
// off, then dead, the slave reset, both released; traffic; off and dead together mid-burst; the
// slave reset, released; 30 times off and back with the slave running; traffic; the master stops
// and every burst must complete. Checked: every
// burst answered once, in order; reads ARLEN + 1 beats with RLAST on the last; OKAY reads carry the
// slave's data; a burst taken and answered while neither held is OKAY; a read or a write (its data
// not ahead) taken while off is SLVERR; no hang. Prints PASS, or the errors and FAIL.
module tb_axi_off;
  localparam int AW = 31, DW = 128, NB = 8192;
  logic clk = 1'b0;
  always #5 clk = !clk;
  logic rst = 1'b1, off = 1'b0, dead = 1'b0, frozen = 1'b0, srst = 1'b0;
  logic stop_issue = 1'b0;
  int cyc = 0, errors = 0;
  always_ff @(posedge clk) cyc <= cyc + 1;
  function automatic void fail(string s);
    errors++;
    if (errors <= 20) $display("ERROR at %0d: %s", cyc, s);
  endfunction
  function automatic logic [DW-1:0] sdata(logic [AW-1:0] a, int beat);
    return {4{32'(a) ^ 32'(beat * 32'h9E37_79B9)}};
  endfunction

  // ---- the master's side
  logic [AW-1:0] m_awaddr, m_araddr;  logic [7:0] m_awlen, m_arlen;
  logic m_awvalid = 0, m_awready, m_wvalid = 0, m_wready, m_wlast, m_bvalid, m_bready;
  logic m_arvalid = 0, m_arready, m_rvalid, m_rready, m_rlast;
  logic [DW-1:0] m_wdata, m_rdata;  logic [1:0] m_bresp, m_rresp;
  // ---- the slave's side
  logic [AW-1:0] s_awaddr, s_araddr;  logic [7:0] s_awlen, s_arlen;
  logic s_awvalid, s_awready, s_wvalid, s_wready, s_wlast, s_bvalid, s_bready;
  logic s_arvalid, s_arready, s_rvalid, s_rready, s_rlast;
  logic [DW-1:0] s_wdata, s_rdata;  logic [DW/8-1:0] s_wstrb;  logic [1:0] s_bresp, s_rresp;

  otpu_axi_off #(.AW(AW), .DW(DW), .NOUT(16)) dut (
    .clk, .rst, .off, .dead,
    .m_awaddr, .m_awlen, .m_awvalid, .m_awready, .m_wdata, .m_wstrb('1), .m_wlast, .m_wvalid,
    .m_wready, .m_bresp, .m_bvalid, .m_bready, .m_araddr, .m_arlen, .m_arvalid, .m_arready,
    .m_rdata, .m_rresp, .m_rlast, .m_rvalid, .m_rready,
    .s_awaddr, .s_awlen, .s_awvalid, .s_awready, .s_wdata, .s_wstrb, .s_wlast, .s_wvalid,
    .s_wready, .s_bresp, .s_bvalid, .s_bready, .s_araddr, .s_arlen, .s_arvalid, .s_arready,
    .s_rdata, .s_rresp, .s_rlast, .s_rvalid, .s_rready);

  // ---- the master: planned bursts, AW / W / AR independent, B / R checked in order
  logic [7:0]    w_len [NB], r_len [NB];
  logic [AW-1:0] r_addr [NB];
  bit            w_ahead [NB], w_on_ok [NB], r_on_ok [NB], w_off_hs [NB], r_off_hs [NB];
  int n_w = 0, aw_i = 0, w_i = 0, w_beat = 0, b_i = 0;
  int n_r = 0, ar_i = 0, r_i = 0, r_beat = 0;
  int n_okay = 0, n_err = 0;
  initial for (int k = 0; k < NB; k++) begin
    w_len[k] = 8'($urandom_range(0, 15)); r_len[k] = 8'($urandom_range(0, 15));
    r_addr[k] = AW'($urandom) & ~AW'(15);
    w_on_ok[k] = 1; r_on_ok[k] = 1;
  end
  wire held = off || dead;
  assign m_awlen = w_len[aw_i];
  assign m_awaddr = AW'(aw_i * 4096);
  assign m_wlast = w_beat == int'(w_len[w_i]);
  assign m_wdata = DW'(w_i * 256 + w_beat);
  assign m_arlen = r_len[ar_i];
  assign m_araddr = r_addr[ar_i];
  always_ff @(posedge clk) begin
    m_bready <= $urandom_range(0, 3) != 0;
    m_rready <= $urandom_range(0, 3) != 0;
    if (rst) begin
      m_awvalid <= 0; m_wvalid <= 0; m_arvalid <= 0;
    end else begin
      // AW (once a burst's data has started, its AW follows even after the master stops)
      if (m_awvalid && m_awready) begin
        if (held) begin w_off_hs[aw_i] <= 1; w_on_ok[aw_i] <= 0; end
        w_ahead[aw_i] <= w_i > aw_i || (w_i == aw_i && w_beat > 0);
        aw_i <= aw_i + 1; m_awvalid <= 0;
      end else if (!m_awvalid && aw_i < NB - 1 && aw_i - b_i < 24 &&
                   ((!stop_issue && $urandom_range(0, 7) == 0) ||
                    w_i > aw_i || (w_i == aw_i && (w_beat > 0 || m_wvalid)))) m_awvalid <= 1;
      // W: its burst's AW taken, or (one in eight) leading it
      if (m_wvalid && m_wready) begin
        if (m_wlast) begin w_i <= w_i + 1; w_beat <= 0; m_wvalid <= w_i + 1 < aw_i; end
        else w_beat <= w_beat + 1;
      end else if (!m_wvalid && $urandom_range(0, 3) == 0) begin
        if (w_i < aw_i) m_wvalid <= 1;
        else if (w_i == aw_i && !stop_issue && w_i < NB - 1 && $urandom_range(0, 2) == 0) m_wvalid <= 1;
      end
      // AR
      if (m_arvalid && m_arready) begin
        if (held) begin r_off_hs[ar_i] <= 1; r_on_ok[ar_i] <= 0; end
        ar_i <= ar_i + 1; m_arvalid <= 0;
      end else if (!m_arvalid && !stop_issue && ar_i < NB - 1 && ar_i - r_i < 24 &&
                   $urandom_range(0, 5) == 0) m_arvalid <= 1;
    end
  end
  // B
  always_ff @(posedge clk) begin
    if (!rst && held) for (int k = b_i; k < aw_i; k++) w_on_ok[k] <= 0;
    if (!rst && m_bvalid && m_bready) begin
      if (b_i >= aw_i) fail($sformatf("B %0d without its AW (%0d taken)", b_i, aw_i));
      if (b_i >= w_i) fail($sformatf("B %0d before its WLAST (%0d bursts' data in)", b_i, w_i));
      if (m_bresp == 2'b00) n_okay++; else n_err++;
      if (w_on_ok[b_i] && !held && m_bresp != 2'b00) fail($sformatf("write %0d SLVERR while on", b_i));
      if (w_off_hs[b_i] && !w_ahead[b_i] && m_bresp == 2'b00) fail($sformatf("write %0d taken while off: OKAY", b_i));
      b_i <= b_i + 1;
    end
  end
  // R
  always_ff @(posedge clk) begin
    if (!rst && held) for (int k = r_i; k < ar_i; k++) r_on_ok[k] <= 0;
    if (!rst && m_rvalid && m_rready) begin
      if (r_i >= ar_i) fail($sformatf("R beat for read %0d not asked (%0d taken)", r_i, ar_i));
      if (m_rlast != (r_beat == int'(r_len[r_i])))
        fail($sformatf("read %0d beat %0d of %0d: RLAST %0d", r_i, r_beat, r_len[r_i] + 1, m_rlast));
      if (m_rresp == 2'b00) begin
        if (r_off_hs[r_i]) fail($sformatf("read %0d taken while off: OKAY", r_i));
        if (m_rdata != sdata(r_addr[r_i], r_beat)) fail($sformatf("read %0d beat %0d: data", r_i, r_beat));
      end else if (r_on_ok[r_i] && !held) fail($sformatf("read %0d SLVERR while on", r_i));
      if (m_rlast) begin r_i <= r_i + 1; r_beat <= 0; n_okay += m_rresp == 2'b00; end
      else r_beat <= r_beat + 1;
    end
  end

  // ---- the slave: in order, random gaps; frozen = a stopped clock (nothing moves); srst = reset
  logic [AW-1:0] q_ra [64];  logic [7:0] q_rl [64];  int q_rw = 0, q_rr = 0, s_rbeat = 0;
  int q_aw = 0, q_wl = 0, q_b = 0, b_wait = 0, s_wbeat = 0;
  logic [7:0] q_awl [64];  int q_wn [64];    // each burst's AWLEN, and the beats its data had
  logic s_bv = 0, s_rv = 0, s_awr = 0, s_wr = 0, s_arr = 0;
  assign s_awready = s_awr; assign s_wready = s_wr; assign s_arready = s_arr;
  assign s_bvalid = s_bv; assign s_bresp = 2'b00;
  assign s_rvalid = s_rv; assign s_rresp = 2'b00;
  assign s_rdata = sdata(q_ra[q_rr % 64], s_rbeat);
  assign s_rlast = s_rbeat == int'(q_rl[q_rr % 64]);
  always_ff @(posedge clk) begin
    if (srst || rst) begin
      q_rw <= 0; q_rr <= 0; s_rbeat <= 0; q_aw <= 0; q_wl <= 0; q_b <= 0; s_wbeat <= 0;
      s_bv <= 0; s_rv <= 0; s_awr <= 0; s_wr <= 0; s_arr <= 0; b_wait <= 0;
    end else if (!frozen) begin
      s_awr <= $urandom_range(0, 2) != 0 && q_aw - q_b < 8;
      s_wr  <= $urandom_range(0, 2) != 0;
      s_arr <= $urandom_range(0, 2) != 0 && q_rw - q_rr < 8;
      if (s_awvalid && s_awready) begin q_awl[q_aw % 64] <= s_awlen; q_aw <= q_aw + 1; end
      // the slave's W in step with its bursts: each beat's number (the master's data, low byte)
      if (s_wvalid && s_wready) begin
        if (int'(s_wdata[7:0]) != s_wbeat) fail($sformatf("slave: W beat %0d of its burst %0d carries beat %0d", s_wbeat, q_wl, s_wdata[7:0]));
        if (s_wlast) begin q_wn[q_wl % 64] <= s_wbeat + 1; q_wl <= q_wl + 1; s_wbeat <= 0; end
        else s_wbeat <= s_wbeat + 1;
      end
      if (s_bv && s_bready && q_wn[q_b % 64] != int'(q_awl[q_b % 64]) + 1)
        fail($sformatf("slave: burst %0d had %0d beats for AWLEN %0d", q_b, q_wn[q_b % 64], q_awl[q_b % 64]));
      if (s_arvalid && s_arready) begin
        q_ra[q_rw % 64] <= s_araddr; q_rl[q_rw % 64] <= s_arlen; q_rw <= q_rw + 1;
      end
      // B: a burst with its AW and its data in, after a few cycles
      if (s_bv && s_bready) begin s_bv <= 0; q_b <= q_b + 1; b_wait <= $urandom_range(2, 12); end
      else if (!s_bv && q_b < q_aw && q_b < q_wl) begin
        if (b_wait == 0) s_bv <= 1; else b_wait <= b_wait - 1;
      end
      // R
      if (s_rv && s_rready) begin
        if (s_rlast) begin q_rr <= q_rr + 1; s_rbeat <= 0; end
        else s_rbeat <= s_rbeat + 1;
        s_rv <= $urandom_range(0, 3) != 0 && (q_rw > q_rr + 1 || !s_rlast);
      end else if (!s_rv && q_rr < q_rw) s_rv <= $urandom_range(0, 3) != 0;
    end
  end

  // ---- the scenario
  task automatic cycles(int n); repeat (n) @(negedge clk); endtask
  task automatic revive();
    cycles(200);
    srst = 1; cycles(4); srst = 0;           // the slave reset (its clock back)
    frozen = 0;
    cycles(20); dead = 0;
    cycles(40); off = 0;
  endtask
  initial begin
    cycles(5); rst = 0;
    cycles(3000);                                          // traffic
    off = 1; cycles(800);                                  // off: the slave finishes its bursts
    if (dut.n_r != 0 && !dut.rg) fail("off: reads not handed to the gate after 800 cycles");
    dead = 1; frozen = 1; revive();
    for (int k = 0; k < 30; k++) begin                     // off and back, the slave alive
      cycles($urandom_range(100, 400)); off = 1;
      cycles($urandom_range(100, 400)); off = 0;
    end
    cycles(3000);                                          // traffic again
    off = 1; dead = 1; frozen = 1; revive();               // all at once, mid-burst
    cycles(3000);
    stop_issue = 1;
    for (int k = 0; k < 40000 && (b_i < aw_i || w_i < aw_i || r_i < ar_i); k++) @(negedge clk);
    if (b_i != aw_i || r_i != ar_i) fail($sformatf("hang: B %0d of %0d, R %0d of %0d (W %0d)", b_i, aw_i, r_i, ar_i, w_i));
    if (aw_i < 200 || ar_i < 200) fail($sformatf("too little traffic: %0d writes, %0d reads", aw_i, ar_i));
    if (n_err == 0) fail("no burst answered by the gate");
    $display("writes %0d, reads %0d: OKAY %0d, SLVERR %0d (beats or bursts)", aw_i, ar_i, n_okay, n_err);
    if (errors == 0) $display("PASS"); else $display("FAIL (%0d errors)", errors);
    $finish;
  end
endmodule
