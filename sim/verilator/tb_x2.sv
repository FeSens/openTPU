// Two BCU1525 boards at S 2 / NC 2, joined by two otpu_link_sim cables. The board / DIMM
// path is tb_bcuwg's (otpu_bcu_wg_sys, bridges, pairs, hmuxes and otpu_dimm_model), replicated
// per card; clocks and calibration are shared. Host protocol: X2SimTransport's @<card> W/R/P,
// global C (all numbers hex), REG lines in script order, DONE at the end. Physical images:
// ch<16 card + 8 slice + channel>.bin / _hi.txt, dumped as _out.bin / _hi_out.txt.
// OTPU_BOARD supplies the per-card architecture, exactly as for tb_bcuwg. L defaults to 40;
// +lnk0_L=N / +lnk1_L=N override it. The host's +link_lat alias is handled by the cable model.
// Phase 1 FOLLOW is local only. Status / hello / INFO are modelled; ping and link counters
// read zero. Cable identities are fixed at 0 / 1, and LRST on either end resets both cables.
module tb_x2;
  parameter int NCARDS = 2;
  parameter int S        = 2;             // slices (otpu_bcu_wg_sys S)
  parameter int NCH      = 8;
  parameter int WORDS    = 1 << 20;       // a slice's logical memory words (its NCH channels)
  parameter int CORE_KHZ = 100000;
  parameter logic [4*S*NCH-1:0] RLY = '0; // channel j's bridge pipeline N (otpu_bcu_wg_sys)
  parameter logic [7:0] DMAP = 8'hE4;     // the DIMM slots' DIMMs (otpu_bcu_wg_sys)
  parameter int MWAIT    = 8;             // otpu_bcu_wg_sys's: a partial burst's wait,
  parameter int ARD      = 32;            // the bridges' read credits,
  parameter int PRD      = 32;            // the pairs' R FIFOs (512-bit beats)
  parameter int CAW      = 22;            // the channels' beat address bits (otpu_bcu_wg_sys)
  parameter int BHW      = 0;             // the pairs' bank-half window (otpu_mig_pair; 0: none)

  localparam int ND = NCH / 2, CB = 128, NC = 2 * S, DAW = CAW + 7 + $clog2(NC);
  // the DIMM slot on DIMM d (its channels NC slot ..)
  function automatic int dimm_slot(input int d);
    for (int i = 0; i < ND; i++) if (int'(DMAP[2 * i +: 2]) == d) return i;
    return 0;
  endfunction
  localparam int BEATS = WORDS / NCH / 8;          // a channel's 32-byte beats
  int cp = 5000, ap = 1666;
  logic cclk = 1'b0, cclk_f = 1'b0, aclk = 1'b0, dump = 1'b0;
  initial begin
    void'($value$plusargs("cp=%d", cp));
    void'($value$plusargs("ap=%d", ap));
  end
  always #(ap) aclk = ~aclk;
  initial forever begin                   // clk_f rises with each edge of the core clock
    #(cp / 2) cclk_f = 1'b0;
    #(cp - cp / 2) begin cclk = ~cclk; cclk_f = 1'b1; end
  end
  logic rst = 1'b1, arst = 1'b1;
  logic [ND-1:0] calib = '0;
  longint cyc = 0, acyc = 0;
  always @(posedge cclk) cyc <= cyc + 1;
  always @(posedge aclk) begin
    acyc <= acyc + 1;
    if (acyc == 40) arst <= 1'b0;
    if (acyc == 200) calib <= '1;
  end

  initial if (NCARDS != 2 || S != 2) $fatal(1, "tb_x2: two cards of two slices required");
  logic [NCARDS-1:0] x2_rst;
  logic [NCARDS-1:0][7:0] x2_card, x2_ncards, x2_c_addr;
  logic [NCARDS-1:0] x2_c_we;
  logic [NCARDS-1:0][31:0] x2_c_wd, x2_c_rd;
  logic [NCARDS-1:0][1:0][255:0] td, rd;
  logic [NCARDS-1:0][1:0] tv, tr, tl, rv, rr, rl, rh, up, rxerr, abin, ab;
  logic [NCARDS-1:0][1:0][31:0] stat, hello;
  for (genvar card = 0; card < NCARDS; card++) begin : g_c
  // ---- the control port (core clock)
  logic [11:0] awaddr = '0, araddr = '0;
  logic [31:0] wdata = '0, rdata;
  logic        awvalid = 1'b0, wvalid = 1'b0, arvalid = 1'b0;
  logic        awready, wready, bvalid, arready, rvalid;
  logic [1:0]  bresp, rresp;
  // ---- the pairs' ports (into the hmuxes)
  logic [ND-1:0]        m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready;
  logic [ND-1:0]        m_arvalid, m_arready, m_rvalid, m_rready, m_rlast;
  logic [ND-1:0][3:0]   m_awid, m_arid, m_bid, m_rid;
  logic [ND-1:0][DAW-1:0] m_awaddr, m_araddr;
  logic [ND-1:0][7:0]   m_awlen, m_arlen;
  logic [ND-1:0][511:0] m_wdata, m_rdata;
  logic [ND-1:0][63:0]  m_wstrb;
  logic [ND-1:0][1:0]   m_bresp, m_rresp;
  // ---- the DIMMs' ports (the hmuxes' m_*)
  logic [ND-1:0]        x_awvalid, x_awready, x_wvalid, x_wready, x_wlast, x_bvalid, x_bready;
  logic [ND-1:0]        x_arvalid, x_arready, x_rvalid, x_rready, x_rlast;
  logic [ND-1:0][3:0]   x_awid, x_arid, x_bid, x_rid;
  logic [ND-1:0][DAW-1:0] x_awaddr, x_araddr;
  logic [ND-1:0][7:0]   x_awlen, x_arlen;
  logic [ND-1:0][511:0] x_wdata, x_rdata;
  logic [ND-1:0][63:0]  x_wstrb;
  logic [ND-1:0][1:0]   x_bresp, x_rresp;
  logic [2:0]           led;

  otpu_bcu_wg_sys #(.NCARDS(NCARDS), .S(S), .NCH(NCH), .CB(CB), .CAW(CAW), .CORE_KHZ(CORE_KHZ), .RLY(RLY),
                   .DMAP(DMAP), .MWAIT(MWAIT), .ARD(ARD), .PRD(PRD), .BHW(BHW)) dut (
    .clk(cclk), .clk_f(cclk_f), .rst, .ctl_mrst(1'b0), .aclk, .arst({ND{arst}}), .calib,
    .temp(12'd0), .led,
    .x2_s_tdata(td[card]), .x2_s_tvalid(tv[card]), .x2_s_tlast(tl[card]), .x2_s_tready(tr[card]),
    .x2_m_tdata(rd[card]), .x2_m_tvalid(rv[card]), .x2_m_tlast(rl[card]), .x2_m_thdr(rh[card]),
    .x2_m_tready(rr[card]), .x2_up(up[card]), .x2_rx_err(rxerr[card]),
    .x2_abort_in(abin[card]), .x2_abort(ab[card]), .x2_rst(x2_rst[card]),
    .x2_card(x2_card[card]), .x2_ncards(x2_ncards[card]), .x2_c_we(x2_c_we[card]),
    .x2_c_addr(x2_c_addr[card]), .x2_c_wd(x2_c_wd[card]), .x2_c_rd(x2_c_rd[card]),
    .s_ctl_awaddr(awaddr), .s_ctl_awvalid(awvalid), .s_ctl_awready(awready),
    .s_ctl_wdata(wdata), .s_ctl_wstrb(4'hF), .s_ctl_wvalid(wvalid), .s_ctl_wready(wready),
    .s_ctl_bresp(bresp), .s_ctl_bvalid(bvalid), .s_ctl_bready(1'b1),
    .s_ctl_araddr(araddr), .s_ctl_arvalid(arvalid), .s_ctl_arready(arready),
    .s_ctl_rdata(rdata), .s_ctl_rresp(rresp), .s_ctl_rvalid(rvalid), .s_ctl_rready(1'b1),
    .m_awvalid, .m_awready, .m_awid, .m_awaddr, .m_awlen, .m_wvalid, .m_wready, .m_wdata,
    .m_wstrb, .m_wlast, .m_bvalid, .m_bready, .m_bid, .m_bresp, .m_arvalid, .m_arready, .m_arid,
    .m_araddr, .m_arlen, .m_rvalid, .m_rready, .m_rid, .m_rdata, .m_rresp, .m_rlast);

  // the dump request crosses into aclk as a level the models see for some edges; they write
  // their images at the first
  logic adump = 1'b0, adump_q = 1'b0;
  always @(posedge aclk) begin adump <= dump; adump_q <= adump; end
  int     merr [ND];
  longint mrd [ND], mwr [ND];
  for (genvar d = 0; d < ND; d++) begin : g_dimm
    otpu_mig_hmux #(.IDW(4), .AW(DAW), .HB(S == 1 ? 2 : 0)) u_hmux (.clk(aclk), .rst(arst),
      .p_awvalid(m_awvalid[d]), .p_awready(m_awready[d]), .p_awid(m_awid[d]),
      .p_awaddr(m_awaddr[d]), .p_awlen(m_awlen[d]), .p_wvalid(m_wvalid[d]),
      .p_wready(m_wready[d]), .p_wdata(m_wdata[d]), .p_wstrb(m_wstrb[d]), .p_wlast(m_wlast[d]),
      .p_bvalid(m_bvalid[d]), .p_bready(m_bready[d]), .p_bid(m_bid[d]), .p_bresp(m_bresp[d]),
      .p_arvalid(m_arvalid[d]), .p_arready(m_arready[d]), .p_arid(m_arid[d]),
      .p_araddr(m_araddr[d]), .p_arlen(m_arlen[d]), .p_rvalid(m_rvalid[d]),
      .p_rready(m_rready[d]), .p_rid(m_rid[d]), .p_rdata(m_rdata[d]), .p_rresp(m_rresp[d]),
      .p_rlast(m_rlast[d]),
      .h_awvalid(1'b0), .h_awready(), .h_awaddr('0), .h_awlen('0), .h_wvalid(1'b0), .h_wready(),
      .h_wdata('0), .h_wstrb('0), .h_wlast(1'b0), .h_bvalid(), .h_bready(1'b1), .h_bresp(),
      .h_arvalid(1'b0), .h_arready(), .h_araddr('0), .h_arlen('0), .h_rvalid(), .h_rready(1'b1),
      .h_rdata(), .h_rresp(), .h_rlast(),
      .m_awvalid(x_awvalid[d]), .m_awready(x_awready[d]), .m_awid(x_awid[d]),
      .m_awaddr(x_awaddr[d]), .m_awlen(x_awlen[d]), .m_wvalid(x_wvalid[d]),
      .m_wready(x_wready[d]), .m_wdata(x_wdata[d]), .m_wstrb(x_wstrb[d]), .m_wlast(x_wlast[d]),
      .m_bvalid(x_bvalid[d]), .m_bready(x_bready[d]), .m_bid(x_bid[d]), .m_bresp(x_bresp[d]),
      .m_arvalid(x_arvalid[d]), .m_arready(x_arready[d]), .m_arid(x_arid[d]),
      .m_araddr(x_araddr[d]), .m_arlen(x_arlen[d]), .m_rvalid(x_rvalid[d]),
      .m_rready(x_rready[d]), .m_rid(x_rid[d]), .m_rdata(x_rdata[d]), .m_rresp(x_rresp[d]),
      .m_rlast(x_rlast[d]));
    otpu_dimm_model #(.BEATS(BEATS), .DAW(DAW), .DIMM(d), .NC(NC), .CH0(NCH * S * card + NC * dimm_slot(d)),
                      .BHW(BHW)) u_dimm (
      .clk(aclk), .rst(arst),
      .dump(adump && !adump_q),
      .awvalid(x_awvalid[d]), .awready(x_awready[d]), .awid(x_awid[d]), .awaddr(x_awaddr[d]),
      .awlen(x_awlen[d]), .wvalid(x_wvalid[d]), .wready(x_wready[d]), .wdata(x_wdata[d]),
      .wstrb(x_wstrb[d]), .wlast(x_wlast[d]), .bvalid(x_bvalid[d]), .bready(x_bready[d]),
      .bid(x_bid[d]), .bresp(x_bresp[d]), .arvalid(x_arvalid[d]), .arready(x_arready[d]),
      .arid(x_arid[d]), .araddr(x_araddr[d]), .arlen(x_arlen[d]), .rvalid(x_rvalid[d]),
      .rready(x_rready[d]), .rid(x_rid[d]), .rdata(x_rdata[d]), .rresp(x_rresp[d]),
      .rlast(x_rlast[d]), .nrd(mrd[d]), .nwr(mwr[d]), .nerr(merr[d]));
  end

  // Handshakes driven and sampled on the falling edge, as tb_board's
  task automatic lwrite(input logic [11:0] a, input logic [31:0] v);
    @(negedge cclk);
    awaddr = a; wdata = v; awvalid = 1'b1; wvalid = 1'b1;
    #1;
    while (!(awready && wready)) begin
      @(negedge cclk);
      #1;
    end
    @(negedge cclk);
    awvalid = 1'b0; wvalid = 1'b0;
    while (!bvalid) @(negedge cclk);
  endtask
  task automatic lread(input logic [11:0] a, output logic [31:0] v);
    @(negedge cclk);
    araddr = a; arvalid = 1'b1;
    while (!arready) @(negedge cclk);
    @(negedge cclk);
    arvalid = 1'b0;
    while (!rvalid) @(negedge cclk);
    v = rdata;
  endtask

  logic [S*NCH-1:0] bridge_err;
  assign bridge_err = dut.n_err;
  task automatic dimm_check(input string when);
    for (int d = 0; d < ND; d++)
      if (merr[d] != 0) $fatal(1, "tb_x2: %s: DIMM %0d: %0d errors", when, d, merr[d]);
    if (|bridge_err) $fatal(1, "tb_x2: %s: bridge errors %b", when, bridge_err);
  endtask

  end

  // The cable model supplies its status and hello registers. Other link counters / ping are
  // zero in this co-sim; protocol / CRC / credit behavior is covered by test_x2_link.py.
  for (genvar c = 0; c < NCARDS; c++) begin : g_csr
    always_comb begin
      x2_c_rd[c] = '0;
      case (x2_c_addr[c])
        8'h08: x2_c_rd[c] = {16'd2048, 8'd32, 8'd1};
        8'h10: x2_c_rd[c] = stat[c][0];
        8'h14: x2_c_rd[c] = hello[c][0];
        8'h20: x2_c_rd[c] = stat[c][1];
        8'h24: x2_c_rd[c] = hello[c][1];
        default: ;
      endcase
    end
  end
  for (genvar p = 0; p < 2; p++) begin : g_k
    otpu_link_sim #(.ID(p), .CA(0), .PA(p), .CB(1), .PB(p), .NC(NCARDS)) u_link (
      .clk(cclk), .rst(rst || |x2_rst),
      .a_s_tdata(td[0][p]), .a_s_tvalid(tv[0][p]), .a_s_tready(tr[0][p]), .a_s_tlast(tl[0][p]),
      .a_m_tdata(rd[0][p]), .a_m_tvalid(rv[0][p]), .a_m_tready(rr[0][p]),
      .a_m_tlast(rl[0][p]), .a_m_thdr(rh[0][p]), .a_up(up[0][p]), .a_rx_err(rxerr[0][p]),
      .a_abort_in(abin[0][p]), .a_abort(ab[0][p]), .a_stat(stat[0][p]), .a_hello(hello[0][p]),
      .b_s_tdata(td[1][p]), .b_s_tvalid(tv[1][p]), .b_s_tready(tr[1][p]), .b_s_tlast(tl[1][p]),
      .b_m_tdata(rd[1][p]), .b_m_tvalid(rv[1][p]), .b_m_tready(rr[1][p]),
      .b_m_tlast(rl[1][p]), .b_m_thdr(rh[1][p]), .b_up(up[1][p]), .b_rx_err(rxerr[1][p]),
      .b_abort_in(abin[1][p]), .b_abort(ab[1][p]), .b_stat(stat[1][p]), .b_hello(hello[1][p]));
  end
  task automatic lwrite(input int c, input logic [11:0] a, input logic [31:0] v);
    if (c == 0) g_c[0].lwrite(a, v); else g_c[1].lwrite(a, v);
  endtask
  task automatic lread(input int c, input logic [11:0] a, output logic [31:0] v);
    if (c == 0) g_c[0].lread(a, v); else g_c[1].lread(a, v);
  endtask
  string dir, op;
  integer fd, n, card;
  longint max_cycles = 64'd1 << 40;
  integer debug_fd = 0;
  always @(posedge cclk) begin
    if (debug_fd != 0 && cyc % 1024 == 0) begin
      $fdisplay(debug_fd, "cycle=%0d card0 run=%b load=%b halt=%b err=%b coll=%h seq=%0d; card1 run=%b load=%b halt=%b err=%b coll=%h seq=%0d",
                cyc, g_c[0].dut.u_board.run, g_c[0].dut.u_board.ld_busy,
                g_c[0].dut.u_board.halted, g_c[0].dut.u_board.error,
                g_c[0].dut.u_board.coll_status, g_c[0].dut.u_board.x2_seq,
                g_c[1].dut.u_board.run, g_c[1].dut.u_board.ld_busy,
                g_c[1].dut.u_board.halted, g_c[1].dut.u_board.error,
                g_c[1].dut.u_board.coll_status, g_c[1].dut.u_board.x2_seq);
      $fflush(debug_fd);
    end
    if (cyc > max_cycles) $fatal(1, "tb_x2: timeout at cycle %0d", cyc);
  end
  initial begin
    logic [31:0] a, m, v, r;
    if (!$value$plusargs("dir=%s", dir)) $fatal(1, "+dir= missing");
    void'($value$plusargs("max_cycles=%d", max_cycles));
    if ($test$plusargs("x2_debug")) debug_fd = $fopen($sformatf("%s/x2_progress.log", dir), "w");
    repeat (4) @(negedge cclk); rst = 1'b0;
    while (calib != '1) @(negedge cclk);
    repeat (16) @(negedge cclk);
    fd = $fopen($sformatf("%s/host.txt", dir), "r");
    if (fd == 0) $fatal(1, "no host script");
    while (!$feof(fd)) begin
      n = $fscanf(fd, "%s", op);
      if (n != 1) break;
      if (op == "C") begin
        n = $fscanf(fd, "%h", v);
        repeat (v) @(posedge cclk);
      end else begin
        if ($sscanf(op, "@%d", card) != 1 || card < 0 || card >= NCARDS)
          $fatal(1, "bad card selector %s", op);
        n = $fscanf(fd, "%s", op);
        case (op)
          "W": begin n = $fscanf(fd, "%h %h", a, v); lwrite(card, a[11:0], v); end
          "R": begin
            n = $fscanf(fd, "%h", a); lread(card, a[11:0], r);
            $display("REG %h %h", a, r);
          end
          "P": begin
            n = $fscanf(fd, "%h %h %h", a, m, v);
            do begin
              lread(card, a[11:0], r);
              if (cyc > max_cycles) $fatal(1, "TIMEOUT polling card %0d %h (%h)", card, a, r);
            end while ((r & m) != v);
          end
          default: $fatal(1, "bad host op %s", op);
        endcase
      end
    end
    $fclose(fd);
    @(negedge cclk); dump = 1'b1;
    repeat (4) @(negedge cclk);
    dump = 1'b0;
    repeat (4) @(negedge cclk);
    g_c[0].dimm_check("done"); g_c[1].dimm_check("done");
    if (debug_fd != 0) $fclose(debug_fd);
    $display("DONE cycles=%0d", cyc); $finish;
  end
endmodule
