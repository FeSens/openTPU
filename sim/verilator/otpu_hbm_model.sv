// A behavioural model of one AXI3 port of a memory controller (simulation only: tb_hbmch):
// 256-bit beats (AxSIZE 32 bytes, INCR), a memory of BEATS beats at the port's
// pseudo-channel (byte addresses PC x 2^29 + 0 .. BEATS x 32 - 1), and the freedom AXI gives the
// controller between its read and write channels: a read takes its data at a random cycle between
// its AR and its first R beat, a write lands at a random cycle between its last W beat and its B,
// so a read and a write in flight together on one address may pass each other either way. Writes
// land in AW order and their B's come in that order; reads return in AR order, whatever their IDs
// (BID and RID echo AWID and ARID: in order is one of the orders AXI allows between IDs).
//
// Timing (aclk cycles): +hbm_lat=N from an AR to its first R beat (default 126: 280 ns at 450 MHz,
// the probe's figure), +hbm_wlat=N from a last W beat to its B (77: 172 ns), +hbm_jit=N up to N
// more on each, at random; +hbm_rbw=P: the R channel carries a beat in P percent of its cycles at
// most (the refresh / turnaround share; default 100); +hbm_rgap=P: the R channel holds its next
// beat in P percent of its cycles at random (gaps inside a burst; default 0); +hbm_ast=P /
// +hbm_wst=P: ARREADY / AWREADY low, WREADY low in P percent of the cycles. ARQ read bursts and AWQ write bursts in flight at
// most (AR / AW held off beyond).
//
// Checks: a valid held until its ready with the same payload (AR, AW, W); addresses 32-byte
// aligned, on the port's pseudo-channel, in the modelled memory, not crossing 4 KiB; WLAST on the
// burst's last beat; W only after its AW. wap: the W beats landed since reset (tb_hbmch checks
// n_wdone against it). Prints the R channel's beats per cycle between its first AR and its last
// R beat at the end.
//
// PHYS 1: the memory is board channel PC's physical image, as otpu_native_mem's:
// <d>/ch<PC>.bin (+dir=<d>; 32-bit big-endian words, as $fread reads them) in at time 0,
// <d>/ch<PC>_out.bin (little-endian words) out at the rising edge that sees dump (what has landed
// by then; the bursts that have not are counted in a message); byte o of the image at the port's
// byte address o.
module otpu_hbm_model #(
  parameter int BEATS = 1 << 16,
  parameter logic [4:0] PC = 5'd0,
  parameter int ARQ = 64,
  parameter int AWQ = 32,
  parameter bit PHYS = 1'b0
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         dump,
  input  logic         awvalid,
  output logic         awready,
  input  logic [5:0]   awid,
  input  logic [33:0]  awaddr,
  input  logic [3:0]   awlen,
  input  logic         wvalid,
  output logic         wready,
  input  logic [255:0] wdata,
  input  logic [31:0]  wstrb,
  input  logic         wlast,
  output logic         bvalid,
  input  logic         bready,
  output logic [5:0]   bid,
  output logic [1:0]   bresp,
  input  logic         arvalid,
  output logic         arready,
  input  logic [5:0]   arid,
  input  logic [33:0]  araddr,
  input  logic [3:0]   arlen,
  output logic         rvalid,
  input  logic         rready,
  output logic [5:0]   rid,
  output logic [255:0] rdata,
  output logic [1:0]   rresp,
  output logic         rlast,
  output longint       wap,
  output int           nerr
);
  typedef struct { int b; int n; logic [5:0] id; longint ts, tr; bit got; logic [15:0][255:0] d; } rd_t;
  typedef struct { int b; int n; logic [5:0] id; } aw_t;
  typedef struct { int b; int n; logic [5:0] id; longint ta, tb; bit done; logic [15:0][255:0] d; logic [15:0][31:0] s; } wr_t;
  logic [255:0] mem [BEATS];
  rd_t    rq [$];
  aw_t    awq [$];
  wr_t    wq [$];
  wr_t    wcur;
  int     wk = 0;                          // W beats of the head AW taken
  int     rk = 0;                          // R beats of the head read given
  int     lat = 126, wlat = 77, jit = 0, rbw = 100, rgap = 0, ast = 0, wst = 0;
  longint cyc = 0, ta_last = 0, tb_last = 0, tr_last = 0, r0 = -1, r1 = 0, nr = 0, nw = 0;
  int     tok = 0;
  logic   p_arv = 1'b0, p_awv = 1'b0, p_wv = 1'b0, p_arr = 1'b0, p_awr = 1'b0, p_wr = 1'b0;
  logic [33:0] p_ara, p_awa;
  logic [3:0]  p_arl, p_awl;
  logic [255:0] p_wd;
  logic [31:0] p_ws;
  logic        p_wl;
  initial begin
    void'($value$plusargs("hbm_lat=%d", lat));
    void'($value$plusargs("hbm_wlat=%d", wlat));
    void'($value$plusargs("hbm_jit=%d", jit));
    void'($value$plusargs("hbm_rbw=%d", rbw));
    void'($value$plusargs("hbm_rgap=%d", rgap));
    void'($value$plusargs("hbm_ast=%d", ast));
    void'($value$plusargs("hbm_wst=%d", wst));
    for (int i = 0; i < BEATS; i++) mem[i] = '0;
    if (PHYS) begin
      string dir;
      int fd, nread;
      logic [31:0] w [BEATS * 8];
      if (!$value$plusargs("dir=%s", dir)) $fatal(1, "otpu_hbm_model: PHYS 1 needs +dir=");
      for (int i = 0; i < BEATS * 8; i++) w[i] = '0;
      fd = $fopen($sformatf("%s/ch%0d.bin", dir, PC), "rb");
      if (fd != 0) begin
        nread = $fread(w, fd);
        $fclose(fd);
      end
      for (int i = 0; i < BEATS; i++)
        for (int k = 0; k < 8; k++) mem[i][32 * k +: 32] = w[8 * i + k];
    end
    nerr = 0; wap = 0;
    arready = 1'b0; awready = 1'b0; wready = 1'b0; rvalid = 1'b0; bvalid = 1'b0;
    rresp = 2'b00; bresp = 2'b00; rlast = 1'b0; rdata = '0; rid = '0; bid = '0;
  end

  function automatic int rnd(input int n);   // 0 .. n
    return n > 0 ? int'($urandom % (n + 1)) : 0;
  endfunction
  // a burst's first beat in the memory, after the address checks
  function automatic int chk(input string ch, input logic [33:0] a, input logic [3:0] l);
    longint off;
    off = longint'(a[28:0]);
    if (a[33:29] != PC || a[4:0] != 5'd0 || off + 32 * (longint'(l) + 1) > longint'(BEATS) * 32 ||
        (off % 4096) + 32 * (longint'(l) + 1) > 4096) begin
      if (nerr < 10) $display("ERROR hbm %0d: %s address %h len %0d", PC, ch, a, l);
      nerr++;
      return 0;
    end
    return int'(off / 32);
  endfunction

  always @(posedge clk) begin
    cyc++;
    if (rst) begin
      rq.delete(); awq.delete(); wq.delete(); wk = 0; rk = 0;
      arready <= 1'b0; awready <= 1'b0; wready <= 1'b0; rvalid <= 1'b0; bvalid <= 1'b0;
      p_arv = 1'b0; p_awv = 1'b0; p_wv = 1'b0; wap = 0;
    end else begin
      // ---- the master's side of the handshakes: held valids keep their payload
      if (p_arv && !p_arr && (!arvalid || araddr != p_ara || arlen != p_arl)) begin
        $display("ERROR hbm %0d: AR dropped or changed before ARREADY", PC); nerr++;
      end
      if (p_awv && !p_awr && (!awvalid || awaddr != p_awa || awlen != p_awl)) begin
        $display("ERROR hbm %0d: AW dropped or changed before AWREADY", PC); nerr++;
      end
      if (p_wv && !p_wr && (!wvalid || wdata != p_wd || wstrb != p_ws || wlast != p_wl)) begin
        $display("ERROR hbm %0d: W dropped or changed before WREADY", PC); nerr++;
      end
      p_arv = arvalid; p_ara = araddr; p_arl = arlen; p_arr = arready;
      p_awv = awvalid; p_awa = awaddr; p_awl = awlen; p_awr = awready;
      p_wv = wvalid; p_wd = wdata; p_ws = wstrb; p_wl = wlast; p_wr = wready;

      // ---- AR: a read takes its data at ts, gives its first beat at tr (in AR order)
      if (arvalid && arready) begin
        rd_t r;
        r.b = chk("AR", araddr, arlen); r.n = int'(arlen) + 1; r.id = arid;
        r.tr = cyc + lat + rnd(jit);
        if (r.tr < tr_last) r.tr = tr_last;
        tr_last = r.tr;
        r.ts = cyc + rnd(int'(r.tr - cyc));
        r.got = 1'b0;
        rq.push_back(r);
        if (r0 < 0) r0 = cyc;
      end
      foreach (rq[i])
        if (!rq[i].got && rq[i].ts <= cyc) begin
          for (int k = 0; k < rq[i].n; k++) rq[i].d[k] = mem[rq[i].b + k];
          rq[i].got = 1'b1;
        end
      // ---- R: the head read's beats, one a cycle at most, rbw percent of the cycles
      tok += rbw;
      if (rvalid && rready) begin
        nr++; r1 = cyc;
        rk++;
        if (rk == rq[0].n) begin void'(rq.pop_front()); rk = 0; end
      end
      if (!rvalid || rready) begin
        if (rq.size() != 0 && rq[0].got && rq[0].tr <= cyc && tok >= 100 &&
            (rgap == 0 || $urandom % 100 >= rgap)) begin
          tok -= 100;
          rvalid <= 1'b1; rdata <= rq[0].d[rk]; rlast <= rk == rq[0].n - 1; rresp <= 2'b00;
          rid <= rq[0].id;
        end else rvalid <= 1'b0;
      end
      if (tok > 100) tok = 100;

      // ---- AW, W: a burst's beats after its AW; once all in, it lands at ta and answers at tb
      if (awvalid && awready) begin
        aw_t w;
        w.b = chk("AW", awaddr, awlen); w.n = int'(awlen) + 1; w.id = awid;
        awq.push_back(w);
      end
      if (wvalid && wready) begin
        if (awq.size() == 0) begin $display("ERROR hbm %0d: W before its AW", PC); nerr++; end
        else begin
          wcur.d[wk] = wdata; wcur.s[wk] = wstrb;
          wk++; nw++;
          if (wlast != (wk == awq[0].n)) begin
            $display("ERROR hbm %0d: WLAST %b at beat %0d of %0d", PC, wlast, wk, awq[0].n); nerr++;
          end
          if (wlast || wk == awq[0].n) begin
            wcur.b = awq[0].b; wcur.n = awq[0].n; wcur.id = awq[0].id;
            wcur.tb = cyc + wlat + rnd(jit);
            if (wcur.tb < tb_last) wcur.tb = tb_last;
            tb_last = wcur.tb;
            wcur.ta = cyc + rnd(int'(wcur.tb - cyc));
            if (wcur.ta < ta_last) wcur.ta = ta_last;
            ta_last = wcur.ta;
            wcur.done = 1'b0;
            wq.push_back(wcur);
            void'(awq.pop_front()); wk = 0;
          end
        end
      end
      foreach (wq[i])
        if (!wq[i].done && wq[i].ta <= cyc) begin
          for (int k = 0; k < wq[i].n; k++)
            for (int j = 0; j < 32; j++)
              if (wq[i].s[k][j]) mem[wq[i].b + k][8 * j +: 8] = wq[i].d[k][8 * j +: 8];
          wq[i].done = 1'b1;
          wap += longint'(wq[i].n);
        end
      // ---- B: in AW order, once landed
      if (bvalid && bready) void'(wq.pop_front());
      if (!bvalid || bready) begin
        if (wq.size() != 0 && wq[0].done && wq[0].tb <= cyc) begin
          bvalid <= 1'b1; bresp <= 2'b00; bid <= wq[0].id;
        end else bvalid <= 1'b0;
      end

      arready <= rq.size() < ARQ && rnd(99) >= ast;
      awready <= awq.size() + wq.size() < AWQ && rnd(99) >= ast;
      wready  <= awq.size() != 0 && rnd(99) >= wst;
    end
  end

  if (PHYS) begin : g_dump
    always @(posedge clk) if (dump) begin
      string dir;
      int fd;
      void'($value$plusargs("dir=%s", dir));
      if (awq.size() != 0 || wq.size() != 0)
        $display("hbm %0d: %0d write bursts not landed at the dump", PC, awq.size() + wq.size());
      fd = $fopen($sformatf("%s/ch%0d_out.bin", dir, PC), "wb");
      for (int i = 0; i < BEATS; i++)
        for (int k = 0; k < 8; k++) $fwrite(fd, "%u", mem[i][32 * k +: 32]);
      $fclose(fd);
    end
  end

  final if (!PHYS)
    $display("hbm %0d: %0d R beats, %0d W beats, %.3f R beats per cycle", PC, nr, nw,
             real'(nr) / real'(r1 - r0 + 1));
endmodule
