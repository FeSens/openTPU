// Debug monitors for XDMA's DMA (otpu_native_sys XMON; a debug build: run_vivado.sh XMON=1), made
// for the card fault of 2026-10-01: with host->card and card->host DMA overlapped, the host's
// write data landed 64 bytes off its addresses for good (both channels, until a reload), or the
// host->card engine stopped (BUSY, no descriptor completed). Passive: it watches XDMA's DMA master
// where it meets otpu_axi_split2 (xclk) and channel 0's two controller ports (uclk), and drives
// nothing there.
//
// The host writes self-describing data (opentpu/host/xmon.py): every 16-byte beat holds the words
// {MAGIC, its byte address as on XDMA's master (bit 31: the channel), that address inverted, a
// tag}. A beat with MAGIC and the inverted copy is checked; other data (the accelerator's, a
// program load) is not:
//   - XDMA's master: each W beat against its AW's address plus 16 bytes per beat before it in the
//     burst, each R beat the same against its AR's (the split answers in AR order); WLAST and
//     RLAST against the lengths. Watchdogs, 2^TW xclk cycles without progress: writes (an AW or W
//     waiting, or a burst whose last beat has not come), B (bursts done without their response),
//     reads (an AR or R waiting, or a read whose last beat has not come). The first stall records
//     the handshake signals and what was open.
//   - Channel 0's controller ports (LiteDRAM's native ports): per port, the write data beats
//     against the write commands, paired in order whichever comes first (the ECC port's register
//     takes a beat before the controller takes its command), and the read data against the read
//     commands. A 64-byte beat holds four 16-byte ones: the checked ones must agree on the beat
//     and the beat must be the command's.
// A fault upstream of the bridges (in XDMA) shows on XDMA's master and on the controller side
// alike; one in otpu_mem_ch on the controller side only; one in the controller or the DRAM in the
// read checks only. A host->card engine that stops with nothing open on its master (an AW never
// issued) shows no stall: then its counts balance.
//
// Registers (otpu_ctrl 0xF00 + 4k: word[k]). Writing 0xF00: bit0 SNAP (copy the counters and
// records of both clocks into their shadows), bit1 CLEAR (zero counters, records and flags). Word
// 0 is live; words 2..31 are the shadows of the last SNAP: read them once word 1's two
// acknowledgement counts equal its request count.
//    0 FLAGS   [31:16] 0x584D ("XM"); sticky: bit0 X_WSHIFT, bit1 X_RSHIFT, bit2 X_WLAST, bit3
//              X_RLAST, bit4 X_WSTALL, bit5 X_BSTALL, bit6 X_RSTALL, bit7 X_PROTO (an address
//              queue overflowed, or a beat came with no address), bit8 N_WSHIFT, bit9 N_RSHIFT,
//              bit10 N_PROTO (a pairing queue overflowed)
//    1 SNAPS   [7:0] SNAP requests, [15:8] acknowledged in xclk, [23:16] in uclk, [31:24] TW
//    2..6      X_AW, X_W, X_B, X_AR, X_R: handshakes (W and R: beats)
//    7, 8      X_WCHK, X_RCHK: self-describing W / R beats checked
//    9, 10     X_WBAD, X_RBAD: of those, off their address
//   11..13     X_WEXP, X_WGOT, X_WTAG: the first bad W beat's address, the address and tag it held
//   14, 15     X_REXP, X_RGOT: the first bad R beat's
//   16         X_STALL (first stall): [11:0] {rlast, rready, rvalid, arready, arvalid, bready,
//              bvalid, wlast, wready, wvalid, awready, awvalid}, [14:12] the watchdogs that had
//              fired {R, B, W}, [15] recorded, [23:16] the open W burst's beat, [31:24] the open
//              R burst's
//   17         X_STALLN (first stall): [7:0] AWs whose last W beat has not come, [15:8] bursts
//              without B, [23:16] ARs whose last R beat has not come, [31:24] the open W burst's
//              AWLEN
//   18, 19     X_LIVE, X_LIVEN: the same at the SNAP ([14:12]: the watchdogs firing then)
//   20..23     N_WCMD, N_WDAT, N_RCMD, N_RDAT: channel 0's write / read commands and data beats
//   24, 25     N_WCHK, N_RCHK: 64-byte beats with a self-describing part, checked
//   26, 27     N_WBAD, N_RBAD: of those, off their command's beat (or disagreeing inside)
//   28, 29     N_WEXP, N_WGOT: the first bad write beat: its command's byte address (bit0: the
//              port) and the beat its data described (bit0: its parts disagreed)
//   30, 31     N_REXP, N_RGOT: the first bad read beat's
// Clock crossings: the requests and acknowledgements as toggles, the flags bit by bit, all through
// ASYNC_REG pairs; the shadows are read in clk only after the acknowledgement crossed
// (constraints/otpu_xmon.tcl). With OTPU_ILA defined (create_project.tcl at XMON) two ILAs
// (otpu_ila_x in xclk, otpu_ila_n in uclk) take the registered signals and the check events.
module otpu_xmon #(
  parameter int TW = 26,                     // watchdog: 2^TW xclk cycles (0.54 s at 125 MHz)
  parameter logic [31:0] MAGIC = 32'h584D_4F4E
) (
  // the register side (otpu_ctrl, core clock)
  input  logic        clk,
  input  logic        snap,
  input  logic        clr,
  output logic [31:0] word [32],
  // XDMA's DMA master as it meets otpu_axi_split2 (xclk)
  input  logic        xclk,
  input  logic        xrst,
  input  logic        x_awvalid,
  input  logic        x_awready,
  input  logic [3:0]  x_awid,
  input  logic [31:0] x_awaddr,
  input  logic [7:0]  x_awlen,
  input  logic        x_wvalid,
  input  logic        x_wready,
  input  logic [127:0] x_wdata,
  input  logic        x_wlast,
  input  logic        x_bvalid,
  input  logic        x_bready,
  input  logic [3:0]  x_bid,
  input  logic        x_arvalid,
  input  logic        x_arready,
  input  logic [3:0]  x_arid,
  input  logic [31:0] x_araddr,
  input  logic [7:0]  x_arlen,
  input  logic        x_rvalid,
  input  logic        x_rready,
  input  logic [3:0]  x_rid,
  input  logic [127:0] x_rdata,
  input  logic        x_rlast,
  // channel 0's controller ports (uclk), [port]
  input  logic        uclk,
  input  logic        urst,
  input  logic [1:0]  c_cmd_valid,
  input  logic [1:0]  c_cmd_ready,
  input  logic [1:0]  c_cmd_we,
  input  logic [1:0][24:0] c_cmd_addr,
  input  logic [1:0]  c_wdata_valid,
  input  logic [1:0]  c_wdata_ready,
  input  logic [1:0][511:0] c_wdata_data,
  input  logic [1:0]  c_rdata_valid,
  input  logic [1:0][511:0] c_rdata_data
);
  // ================================================================ register side (clk)
  logic       t_sn = 1'b0, t_cl = 1'b0;            // SNAP / CLEAR requests, as toggles
  logic       x_ack = 1'b0, u_ack = 1'b0;          // their acknowledgements (xclk / uclk)
  logic [7:0] n_req = '0, n_xa = '0, n_ua = '0;
  (* ASYNC_REG = "TRUE" *) logic [1:0] c_xa = '0, c_ua = '0;
  logic       c_xa3 = 1'b0, c_ua3 = 1'b0;
  logic [7:0] x_flag = '0;                         // (xclk)
  logic [2:0] n_flag = '0;                         // (uclk)
  (* ASYNC_REG = "TRUE" *) logic [7:0] c_xf1 = '0, c_xf2 = '0;
  (* ASYNC_REG = "TRUE" *) logic [2:0] c_nf1 = '0, c_nf2 = '0;
  logic [31:0] sx [2:19];                          // shadows (xclk)
  logic [31:0] sn [20:31];                         // shadows (uclk)
  always_ff @(posedge clk) begin
    if (snap) begin
      t_sn <= !t_sn;
      n_req <= n_req + 1;
    end
    if (clr) t_cl <= !t_cl;
    c_xa <= {c_xa[0], x_ack};
    c_xa3 <= c_xa[1];
    if (c_xa[1] != c_xa3) n_xa <= n_xa + 1;
    c_ua <= {c_ua[0], u_ack};
    c_ua3 <= c_ua[1];
    if (c_ua[1] != c_ua3) n_ua <= n_ua + 1;
    c_xf1 <= x_flag;
    c_xf2 <= c_xf1;
    c_nf1 <= n_flag;
    c_nf2 <= c_nf1;
  end
  assign word[0] = {16'h584D, 5'd0, c_nf2, c_xf2};
  assign word[1] = {8'(TW), n_ua, n_xa, n_req};
  for (genvar k = 2; k < 20; k++) begin : g_wx
    assign word[k] = sx[k];
  end
  for (genvar k = 20; k < 32; k++) begin : g_wn
    assign word[k] = sn[k];
  end

  // a beat (16 bytes) that describes itself
  function automatic logic sd16(input logic [127:0] d);
    return d[31:0] == MAGIC && d[95:64] == ~d[63:32];
  endfunction

  // ================================================================ XDMA's master (xclk)
  (* ASYNC_REG = "TRUE" *) logic [1:0] x_sn = '0, x_cl = '0;
  logic x_sn3 = 1'b0, x_cl3 = 1'b0;
  always_ff @(posedge xclk) begin
    x_sn <= {x_sn[0], t_sn};
    x_sn3 <= x_sn[1];
    x_cl <= {x_cl[0], t_cl};
    x_cl3 <= x_cl[1];
  end
  wire x_snp = x_sn[1] != x_sn3;
  wire x_clp = x_cl[1] != x_cl3;

  // the inputs, registered once (the ILA's probes)
  logic         xr = 1'b1;
  logic         awv, awr, wv, wr, wl, bv, br, arv, arr, rv, rr, rl;
  logic [3:0]   awi, bi, ari, ri;
  logic [31:0]  awa, ara;
  logic [7:0]   awl, arl;
  logic [127:0] wd, rd;
  always_ff @(posedge xclk) begin
    xr <= xrst;
    {awv, awr, wv, wr, wl, bv, br, arv, arr, rv, rr, rl} <=
      {x_awvalid, x_awready, x_wvalid, x_wready, x_wlast, x_bvalid, x_bready,
       x_arvalid, x_arready, x_rvalid, x_rready, x_rlast};
    {awi, bi, ari, ri} <= {x_awid, x_bid, x_arid, x_rid};
    {awa, awl, ara, arl} <= {x_awaddr, x_awlen, x_araddr, x_arlen};
    wd <= x_wdata;
    rd <= x_rdata;
  end
  wire aw_hs = awv && awr && !xr, w_hs = wv && wr && !xr, b_hs = bv && br && !xr;
  wire ar_hs = arv && arr && !xr, r_hs = rv && rr && !xr;

  // W against AW, R against AR
  logic        we_v, we_sd, we_bad, we_lbad, we_pro, re_v, re_sd, re_bad, re_lbad, re_pro;
  logic [31:0] we_exp, we_got, we_tag, re_exp, re_got, re_tag;
  logic [7:0]  w_open, r_open, wk, rk, w_len, r_len;
  otpu_xmon_burst #(.MAGIC(MAGIC)) u_wb (
    .clk(xclk), .rst(xr), .a_hs(aw_hs), .a_addr(awa), .a_len(awl), .d_hs(w_hs), .d_data(wd),
    .d_last(wl), .e_v(we_v), .e_sd(we_sd), .e_bad(we_bad), .e_lbad(we_lbad), .e_pro(we_pro),
    .e_exp(we_exp), .e_got(we_got), .e_tag(we_tag), .n_open(w_open), .k(wk), .len(w_len));
  otpu_xmon_burst #(.MAGIC(MAGIC)) u_rb (
    .clk(xclk), .rst(xr), .a_hs(ar_hs), .a_addr(ara), .a_len(arl), .d_hs(r_hs), .d_data(rd),
    .d_last(rl), .e_v(re_v), .e_sd(re_sd), .e_bad(re_bad), .e_lbad(re_lbad), .e_pro(re_pro),
    .e_exp(re_exp), .e_got(re_got), .e_tag(re_tag), .n_open(r_open), .k(rk), .len(r_len));

  // bursts done without B (B comes after the burst's last W beat)
  logic [7:0] b_pend;
  always_ff @(posedge xclk)
    if (xr) b_pend <= '0;
    else b_pend <= b_pend + 8'(w_hs && wl) - 8'(b_hs);

  // watchdogs: 2^TW cycles with something waiting or open and no progress
  logic [TW:0] wd_w = '0, wd_b = '0, wd_r = '0;
  wire wait_w = awv || wv || w_open != 0, wait_b = bv || b_pend != 0, wait_r = arv || rv || r_open != 0;
  always_ff @(posedge xclk) begin
    if (xr || !wait_w || aw_hs || w_hs) wd_w <= '0; else if (!wd_w[TW]) wd_w <= wd_w + 1;
    if (xr || !wait_b || b_hs) wd_b <= '0; else if (!wd_b[TW]) wd_b <= wd_b + 1;
    if (xr || !wait_r || ar_hs || r_hs) wd_r <= '0; else if (!wd_r[TW]) wd_r <= wd_r + 1;
  end
  localparam logic [TW:0] WD_FIRE = {1'b0, {TW{1'b1}}};
  wire ev_ws = wait_w && !aw_hs && !w_hs && !xr && wd_w == WD_FIRE;
  wire ev_bs = wait_b && !b_hs && !xr && wd_b == WD_FIRE;
  wire ev_rs = wait_r && !ar_hs && !r_hs && !xr && wd_r == WD_FIRE;
  wire [11:0] hsk = {rl, rr, rv, arr, arv, br, bv, wl, wr, wv, awr, awv};
  wire [31:0] st_now = {rk, wk, 1'b1, wd_r[TW], wd_b[TW], wd_w[TW], hsk};
  wire [31:0] stn_now = {w_len, r_open, b_pend, w_open};

  // counters, records, flags; CLEAR zeroes them (not XDMA's reset: they outlive a link reset)
  logic [31:0] x_naw = '0, x_nw = '0, x_nb = '0, x_nar = '0, x_nr = '0;
  logic [31:0] x_wchk = '0, x_rchk = '0, x_wbad = '0, x_rbad = '0;
  logic [31:0] x_wexp = '0, x_wgot = '0, x_wtag = '0, x_rexp = '0, x_rgot = '0;
  logic [31:0] x_st = '0, x_stn = '0;
  logic        x_wrec = 1'b0, x_rrec = 1'b0, x_srec = 1'b0;
  always_ff @(posedge xclk) begin
    if (x_clp) begin
      x_naw <= '0; x_nw <= '0; x_nb <= '0; x_nar <= '0; x_nr <= '0;
      x_wchk <= '0; x_rchk <= '0; x_wbad <= '0; x_rbad <= '0;
      x_wexp <= '0; x_wgot <= '0; x_wtag <= '0; x_rexp <= '0; x_rgot <= '0;
      x_st <= '0; x_stn <= '0;
      x_wrec <= 1'b0; x_rrec <= 1'b0; x_srec <= 1'b0;
      x_flag <= '0;
    end else begin
      x_naw <= x_naw + 32'(aw_hs);
      x_nw  <= x_nw + 32'(w_hs);
      x_nb  <= x_nb + 32'(b_hs);
      x_nar <= x_nar + 32'(ar_hs);
      x_nr  <= x_nr + 32'(r_hs);
      if (we_v && we_sd) begin
        x_wchk <= x_wchk + 1;
        if (we_bad) begin
          x_wbad <= x_wbad + 1;
          x_flag[0] <= 1'b1;
          if (!x_wrec) begin
            x_wrec <= 1'b1;
            x_wexp <= we_exp; x_wgot <= we_got; x_wtag <= we_tag;
          end
        end
      end
      if (re_v && re_sd) begin
        x_rchk <= x_rchk + 1;
        if (re_bad) begin
          x_rbad <= x_rbad + 1;
          x_flag[1] <= 1'b1;
          if (!x_rrec) begin
            x_rrec <= 1'b1;
            x_rexp <= re_exp; x_rgot <= re_got;
          end
        end
      end
      if (we_v && we_lbad) x_flag[2] <= 1'b1;
      if (re_v && re_lbad) x_flag[3] <= 1'b1;
      if (ev_ws) x_flag[4] <= 1'b1;
      if (ev_bs) x_flag[5] <= 1'b1;
      if (ev_rs) x_flag[6] <= 1'b1;
      if ((we_v && we_pro) || (re_v && re_pro)) x_flag[7] <= 1'b1;
      if ((ev_ws || ev_bs || ev_rs) && !x_srec) begin
        x_srec <= 1'b1;
        x_st <= {rk, wk, 1'b1, ev_rs, ev_bs, ev_ws, hsk};
        x_stn <= stn_now;
      end
    end
    if (x_snp) begin
      sx[2] <= x_naw; sx[3] <= x_nw; sx[4] <= x_nb; sx[5] <= x_nar; sx[6] <= x_nr;
      sx[7] <= x_wchk; sx[8] <= x_rchk; sx[9] <= x_wbad; sx[10] <= x_rbad;
      sx[11] <= x_wexp; sx[12] <= x_wgot; sx[13] <= x_wtag; sx[14] <= x_rexp; sx[15] <= x_rgot;
      sx[16] <= x_st; sx[17] <= x_stn; sx[18] <= st_now; sx[19] <= stn_now;
      x_ack <= !x_ack;
    end
  end

  // ================================================================ channel 0's ports (uclk)
  (* ASYNC_REG = "TRUE" *) logic [1:0] u_sn = '0, u_cl = '0;
  logic u_sn3 = 1'b0, u_cl3 = 1'b0;
  always_ff @(posedge uclk) begin
    u_sn <= {u_sn[0], t_sn};
    u_sn3 <= u_sn[1];
    u_cl <= {u_cl[0], t_cl};
    u_cl3 <= u_cl[1];
  end
  wire u_snp = u_sn[1] != u_sn3;
  wire u_clp = u_cl[1] != u_cl3;

  logic              ur = 1'b1;
  logic [1:0]        ncv, ncr, ncwe, nwv, nwr, nrv;
  logic [1:0][24:0]  nca;
  logic [1:0][511:0] nwd, nrd;
  always_ff @(posedge uclk) begin
    ur <= urst;
    {ncv, ncr, ncwe, nwv, nwr, nrv} <= {c_cmd_valid, c_cmd_ready, c_cmd_we, c_wdata_valid,
                                        c_wdata_ready, c_rdata_valid};
    nca <= c_cmd_addr;
    nwd <= c_wdata_data;
    nrd <= c_rdata_data;
  end
  wire [1:0] wc_hs = ncv & ncr & ncwe & {2{!ur}};
  wire [1:0] rc_hs = ncv & ncr & ~ncwe & {2{!ur}};
  wire [1:0] wd_hs = nwv & nwr & {2{!ur}};
  wire [1:0] rd_hs = nrv & {2{!ur}};

  logic [1:0]       pw_v, pw_any, pw_bad, pw_pro, pr_v, pr_any, pr_bad, pr_pro;
  logic [1:0][31:0] pw_exp, pw_got, pr_exp, pr_got;
  for (genvar p = 0; p < 2; p++) begin : g_port
    otpu_xmon_pair #(.MAGIC(MAGIC), .P(p)) u_w (
      .clk(uclk), .rst(ur), .c_hs(wc_hs[p]), .c_addr(nca[p]), .d_hs(wd_hs[p]), .d_data(nwd[p]),
      .p_v(pw_v[p]), .p_any(pw_any[p]), .p_bad(pw_bad[p]), .p_pro(pw_pro[p]),
      .p_exp(pw_exp[p]), .p_got(pw_got[p]));
    otpu_xmon_pair #(.MAGIC(MAGIC), .P(p)) u_r (
      .clk(uclk), .rst(ur), .c_hs(rc_hs[p]), .c_addr(nca[p]), .d_hs(rd_hs[p]), .d_data(nrd[p]),
      .p_v(pr_v[p]), .p_any(pr_any[p]), .p_bad(pr_bad[p]), .p_pro(pr_pro[p]),
      .p_exp(pr_exp[p]), .p_got(pr_got[p]));
  end

  function automatic logic [31:0] cnt2(input logic [1:0] b);
    return 32'(b[0]) + 32'(b[1]);
  endfunction
  logic [31:0] n_wc = '0, n_wd = '0, n_rc = '0, n_rd = '0, n_wchk = '0, n_rchk = '0;
  logic [31:0] n_wbad = '0, n_rbad = '0, n_wexp = '0, n_wgot = '0, n_rexp = '0, n_rgot = '0;
  logic        n_wrec = 1'b0, n_rrec = 1'b0;
  wire  [1:0]  pw_b = pw_v & pw_any & pw_bad, pr_b = pr_v & pr_any & pr_bad;
  always_ff @(posedge uclk) begin
    if (u_clp) begin
      n_wc <= '0; n_wd <= '0; n_rc <= '0; n_rd <= '0; n_wchk <= '0; n_rchk <= '0;
      n_wbad <= '0; n_rbad <= '0; n_wexp <= '0; n_wgot <= '0; n_rexp <= '0; n_rgot <= '0;
      n_wrec <= 1'b0; n_rrec <= 1'b0;
      n_flag <= '0;
    end else begin
      n_wc <= n_wc + cnt2(wc_hs);
      n_wd <= n_wd + cnt2(wd_hs);
      n_rc <= n_rc + cnt2(rc_hs);
      n_rd <= n_rd + cnt2(rd_hs);
      n_wchk <= n_wchk + cnt2(pw_v & pw_any);
      n_rchk <= n_rchk + cnt2(pr_v & pr_any);
      n_wbad <= n_wbad + cnt2(pw_b);
      n_rbad <= n_rbad + cnt2(pr_b);
      if (|pw_b) begin
        n_flag[0] <= 1'b1;
        if (!n_wrec) begin
          n_wrec <= 1'b1;
          n_wexp <= pw_b[0] ? pw_exp[0] : pw_exp[1];
          n_wgot <= pw_b[0] ? pw_got[0] : pw_got[1];
        end
      end
      if (|pr_b) begin
        n_flag[1] <= 1'b1;
        if (!n_rrec) begin
          n_rrec <= 1'b1;
          n_rexp <= pr_b[0] ? pr_exp[0] : pr_exp[1];
          n_rgot <= pr_b[0] ? pr_got[0] : pr_got[1];
        end
      end
      if (|(pw_v & pw_pro) || |(pr_v & pr_pro)) n_flag[2] <= 1'b1;
    end
    if (u_snp) begin
      sn[20] <= n_wc; sn[21] <= n_wd; sn[22] <= n_rc; sn[23] <= n_rd;
      sn[24] <= n_wchk; sn[25] <= n_rchk; sn[26] <= n_wbad; sn[27] <= n_rbad;
      sn[28] <= n_wexp; sn[29] <= n_wgot; sn[30] <= n_rexp; sn[31] <= n_rgot;
      u_ack <= !u_ack;
    end
  end

`ifdef OTPU_ILA
  // the ILAs (create_project.tcl makes them at XMON): the registered signals and the events, each
  // probe a named signal (tools/xmon_ila.tcl finds the probes by these names)
  wire [11:0] ix_hsk = hsk;
  wire [31:0] ix_awaddr = awa, ix_araddr = ara;
  wire [7:0]  ix_awlen = awl, ix_arlen = arl;
  wire [63:0] ix_wtagaddr = {wd[127:96], wd[63:32]}, ix_rtagaddr = {rd[127:96], rd[63:32]};
  wire [15:0] ix_ids = {awi, bi, ari, ri};
  // ix_ev: [7:0] the events (as FLAGS [7:0]), [8] a handshake on any channel (capture
  // qualification: the cycles that moved), [9] on AW or W
  wire [15:0] ix_ev = {6'd0, aw_hs || w_hs, aw_hs || w_hs || b_hs || ar_hs || r_hs,
                       (we_v && we_pro) || (re_v && re_pro), ev_rs, ev_bs, ev_ws, re_v && re_lbad,
                       we_v && we_lbad, re_v && re_sd && re_bad, we_v && we_sd && we_bad};
  otpu_ila_x u_ila_x (
    .clk(xclk), .probe0(ix_hsk), .probe1(ix_awaddr), .probe2(ix_awlen), .probe3(ix_wtagaddr),
    .probe4(ix_araddr), .probe5(ix_arlen), .probe6(ix_rtagaddr), .probe7(ix_ids), .probe8(ix_ev));
  wire [11:0] in_hsk = {nrv, nwr, nwv, ncwe, ncr, ncv};
  wire [49:0] in_addr = {nca[1], nca[0]};
  wire [63:0] in_waddr = {nwd[1][63:32], nwd[0][63:32]}, in_raddr = {nrd[1][63:32], nrd[0][63:32]};
  // in_ev: [1:0] a bad write beat per port, [3:2] a bad read beat, [5:4] / [7:6] write / read
  // queue overflow, [8] a command or data handshake on either port
  wire [15:0] in_ev = {7'd0, |{wc_hs, rc_hs, wd_hs, rd_hs}, pr_v & pr_pro, pw_v & pw_pro, pr_b, pw_b};
  otpu_ila_n u_ila_n (
    .clk(uclk), .probe0(in_hsk), .probe1(in_addr), .probe2(in_waddr), .probe3(in_raddr),
    .probe4(in_ev));
`endif
endmodule

// One direction of XDMA's master: the addresses in order, each data beat checked against its
// address and its place in the burst (results two cycles after the beat). n_open: addresses
// whose last beat has not come; k, len: the open burst's beat and length.
module otpu_xmon_burst #(
  parameter logic [31:0] MAGIC = 32'h0,
  parameter int AQ = 32                       // the split holds at most 16 in flight
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         a_hs,
  input  logic [31:0]  a_addr,
  input  logic [7:0]   a_len,
  input  logic         d_hs,
  input  logic [127:0] d_data,
  input  logic         d_last,
  output logic         e_v,
  output logic         e_sd,
  output logic         e_bad,                 // checked (e_sd) and off its address
  output logic         e_lbad,                // LAST not where the length puts it
  output logic         e_pro,                 // a beat with no address, or the queue overflowed
  output logic [31:0]  e_exp,
  output logic [31:0]  e_got,
  output logic [31:0]  e_tag,
  output logic [7:0]   n_open,
  output logic [7:0]   k,
  output logic [7:0]   len
);
  localparam int QW = $clog2(AQ);
  logic [27:0] qa [AQ];                       // address bits 31:4
  logic [7:0]  ql [AQ];
  logic [QW:0] wp, rp;
  wire  [QW:0] n = wp - rp;
  assign n_open = 8'(n);
  assign len = ql[rp[QW-1:0]];

  logic        s_v, s_sd, s_lbad, s_pro;
  logic [31:0] s_exp, s_got, s_tag;
  always_ff @(posedge clk) begin
    s_v <= 1'b0;
    s_pro <= 1'b0;
    if (rst) begin
      wp <= '0;
      rp <= '0;
      k <= '0;
    end else begin
      if (a_hs) begin
        if (n == (QW+1)'(AQ)) s_pro <= 1'b1;
        else begin
          qa[wp[QW-1:0]] <= a_addr[31:4];
          ql[wp[QW-1:0]] <= a_len;
          wp <= wp + 1;
        end
      end
      if (d_hs) begin
        s_v <= 1'b1;
        s_exp <= {qa[rp[QW-1:0]] + 28'(k), 4'h0};
        s_got <= d_data[63:32];
        s_tag <= d_data[127:96];
        s_sd <= d_data[31:0] == MAGIC && d_data[95:64] == ~d_data[63:32];
        s_lbad <= d_last != (k == ql[rp[QW-1:0]]);
        if (n == 0) s_pro <= 1'b1;
        if (d_last) begin
          k <= '0;
          if (n != 0) rp <= rp + 1;
        end else k <= k + 1;
      end
    end
  end
  always_ff @(posedge clk) begin
    e_v <= s_v || s_pro;
    e_sd <= s_v && s_sd;
    e_bad <= s_exp != s_got;
    e_lbad <= s_v && s_lbad;
    e_pro <= s_pro;
    e_exp <= s_exp;
    e_got <= s_got;
    e_tag <= s_tag;
  end
endmodule

// One port and direction of channel 0's controller port: the commands' beats in order and the data
// beats in order, paired in order whichever comes first; a beat's self-describing 16-byte parts
// must agree on one 64-byte beat, the command's. p_*: a pair (registered).
module otpu_xmon_pair #(
  parameter logic [31:0] MAGIC = 32'h0,
  parameter int P = 0,                        // the port (bit 0 of p_exp)
  parameter int CQ = 64,                      // commands ahead of their data
  parameter int DQ = 4                        // data ahead of its command (the ECC port's register)
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         c_hs,
  input  logic [24:0]  c_addr,
  input  logic         d_hs,
  input  logic [511:0] d_data,
  output logic         p_v,
  output logic         p_any,                 // a part described itself (checked)
  output logic         p_bad,
  output logic         p_pro,                 // a queue overflowed
  output logic [31:0]  p_exp,
  output logic [31:0]  p_got
);
  localparam int CW = $clog2(CQ), DW = $clog2(DQ);
  // the command queue
  logic [24:0] cq [CQ];
  logic [CW:0] cwp, crp;
  wire  [CW:0] cn = cwp - crp;
  // the data: the parts' beat addresses (two stages), then a summary queue {any, agree, beat}
  logic        a_v, b_v;
  logic [3:0]  a_sd;
  logic [3:0][31:0] a_b;
  logic        b_any, b_ok;
  logic [25:0] b_beat;
  logic [27:0] dq [DQ];
  logic [DW:0] dwp, drp;
  wire  [DW:0] dn = dwp - drp;
  wire         pair = cn != 0 && dn != 0;
  logic        c_ovf, d_ovf;

  always_ff @(posedge clk) begin
    a_v <= d_hs && !rst;
    for (int i = 0; i < 4; i++) begin
      a_sd[i] <= d_data[128 * i +: 32] == MAGIC && d_data[128 * i + 64 +: 32] == ~d_data[128 * i + 32 +: 32];
      a_b[i] <= d_data[128 * i + 32 +: 32] - 32'(16 * i);
    end
    b_v <= a_v && !rst;
    b_any <= |a_sd;
    begin
      logic        any, ok;
      logic [25:0] bb;
      any = 1'b0; ok = 1'b1; bb = '0;
      for (int i = 0; i < 4; i++)
        if (a_sd[i]) begin
          if (!any) bb = a_b[i][31:6];
          else if (a_b[i][31:6] != bb) ok = 1'b0;
          if (a_b[i][5:0] != 6'd0) ok = 1'b0;
          any = 1'b1;
        end
      b_ok <= ok;
      b_beat <= bb;
    end
  end

  always_ff @(posedge clk) begin
    c_ovf <= 1'b0;
    d_ovf <= 1'b0;
    if (rst) begin
      cwp <= '0; crp <= '0; dwp <= '0; drp <= '0;
    end else begin
      if (c_hs) begin
        if (cn == (CW+1)'(CQ)) c_ovf <= 1'b1;
        else begin
          cq[cwp[CW-1:0]] <= c_addr;
          cwp <= cwp + 1;
        end
      end
      if (b_v) begin
        if (dn == (DW+1)'(DQ)) d_ovf <= 1'b1;
        else begin
          dq[dwp[DW-1:0]] <= {b_any, b_ok, b_beat};
          dwp <= dwp + 1;
        end
      end
      if (pair) begin
        crp <= crp + 1;
        drp <= drp + 1;
      end
    end
  end

  always_ff @(posedge clk) begin
    logic [27:0] h;
    logic [24:0] c;
    h = dq[drp[DW-1:0]];
    c = cq[crp[CW-1:0]];
    p_v <= (pair || c_ovf || d_ovf) && !rst;
    p_any <= pair && h[27];
    p_bad <= !h[26] || h[25:0] != {1'b0, c};
    p_pro <= c_ovf || d_ovf;
    p_exp <= {1'b0, c, 5'd0, 1'(P)};
    p_got <= {h[25:0], 5'd0, !h[26]};
  end
endmodule
