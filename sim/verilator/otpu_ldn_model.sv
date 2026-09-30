// Simulation model of one channel's two LiteDRAM native ports as LiteDRAMCrossbar serves them
// (two masters; 64-byte beats: 576 bits with ECC, 512 to the user; beat addresses), as otpu_mem_ch
// drives them: port p takes the beats of the banks of parity p (beat bit 7, checked).
// Per port, cmd ready is randomly withheld (+axi_stall=percent) and drops while QD commands are
// held (the bank's command buffer and pipeline), and the port's commands are performed in its
// command order, each 1 to WJ cycles after it was taken or later (default QD 12, WJ 12). The two
// ports share the channel: at most one command is performed per cycle (either port's head, at
// random when both are due), never during a busy stretch (refresh, row changes: +ldn_busy=N, a
// 10-49 cycle pause with probability N per mille per cycle), and the two ports' commands are not
// ordered against each other. A write is performed by asking for its data (the port's wdata
// ready for one cycle) and taking whatever is on the port's wdata, as the crossbar does, without
// looking at valid: data not valid then is an error, and so is any byte not written
// (LiteDRAMNativePortECC rejects partial writes). A read returns the memory as it is when it is
// performed, LAT to LAT + 7 cycles later (+axi_lat=N), on its port, in the order performed, one
// beat a cycle (the channel's data bus), never held back. +ldn_dual=N: the Nth read beat also
// raises the other port's rdata valid (a controller that breaks the contract: otpu_mem_ch's
// n_err).
// With IMG = 1 the channel's memory loads from <dir>/ch<CH>.bin and dumps to <dir>/ch<CH>_out.bin
// on dump.
module otpu_ldn_model #(
  parameter int BEATS = 1 << 12,
  parameter int LAT   = 20,
  parameter int CH    = 0,
  parameter bit IMG   = 1'b0,
  parameter int QD    = 12,
  parameter int WJ    = 12
) (
  input  logic              clk,
  input  logic              rst,
  input  logic [1:0]        c_cmd_valid,
  output logic [1:0]        c_cmd_ready,
  input  logic [1:0]        c_cmd_we,
  input  logic [1:0][24:0]  c_cmd_addr,
  input  logic [1:0]        c_wdata_valid,
  output logic [1:0]        c_wdata_ready,
  input  logic [1:0][511:0] c_wdata_data,
  input  logic [1:0][63:0]  c_wdata_we,
  output logic [1:0]        c_rdata_valid,
  output logic [1:0][511:0] c_rdata_data,
  input  logic              dump
);
  logic [511:0] mem [BEATS];
  int stall = 0, lat = LAT, busy = 2;
  longint cyc = 0, n_rd = 0, n_wr = 0, n_busy = 0, tret = 0, busy_until = 0, n_ret = 0, dual = 0;
  longint texec [2];
  typedef struct { longint t; logic we; logic [24:0] a; } c_t;
  typedef struct { longint t; bit p; logic [511:0] d; } r_t;
  c_t cq0 [$], cq1 [$];
  r_t rq [$];
  logic [24:0] wa [2];                      // the write whose data is asked for this cycle

  always @(posedge clk) begin
    if (rst) begin
      cq0.delete(); cq1.delete(); rq.delete();
      c_cmd_ready <= '0; c_wdata_ready <= '0; c_rdata_valid <= '0;
      texec = '{cyc, cyc}; tret = cyc; busy_until = cyc;
    end else begin
      bit d0, d1, pp;
      c_t h;
      // the write data asked for in the cycle now ending
      for (int p = 0; p < 2; p++)
        if (c_wdata_ready[p]) begin
          if (!c_wdata_valid[p]) $fatal(1, "otpu_ldn_model ch%0d: port %0d write data asked for, not valid", CH, p);
          if (c_wdata_we[p] != '1) $fatal(1, "otpu_ldn_model ch%0d: port %0d partial write (we %h)", CH, p, c_wdata_we[p]);
          mem[wa[p]] = c_wdata_data[p];
          n_wr++;
        end
      // commands taken
      for (int p = 0; p < 2; p++)
        if (c_cmd_valid[p] && c_cmd_ready[p]) begin
          longint t;
          if (c_cmd_addr[p] >= BEATS) $fatal(1, "otpu_ldn_model ch%0d: beat %0d beyond the memory", CH, c_cmd_addr[p]);
          if (c_cmd_addr[p][7] != 1'(p)) $fatal(1, "otpu_ldn_model ch%0d: beat %0d on port %0d", CH, c_cmd_addr[p], p);
          t = cyc + 1 + longint'($urandom % WJ);
          if (t <= texec[p]) t = texec[p] + 1;
          texec[p] = t;
          if (p == 0) cq0.push_back('{t, c_cmd_we[p], c_cmd_addr[p]});
          else cq1.push_back('{t, c_cmd_we[p], c_cmd_addr[p]});
        end
      // one head command performed (the channel's)
      c_wdata_ready <= '0;
      if (busy != 0 && cyc >= busy_until && ($urandom % 1000) < busy) begin
        busy_until = cyc + 10 + longint'($urandom % 40);
        n_busy++;
      end
      d0 = cq0.size() != 0 && cq0[0].t <= cyc;
      d1 = cq1.size() != 0 && cq1[0].t <= cyc;
      if ((d0 || d1) && cyc >= busy_until) begin
        pp = d0 && d1 ? 1'($urandom % 2) : d1;
        h = pp ? cq1[0] : cq0[0];
        if (h.we) begin
          c_wdata_ready[pp] <= 1'b1;
          wa[pp] = h.a;
        end else begin
          longint t;
          t = cyc + lat + longint'($urandom % 8);
          if (t <= tret) t = tret + 1;
          tret = t;
          rq.push_back('{t, pp, mem[h.a]});
          n_rd++;
        end
        if (pp) void'(cq1.pop_front()); else void'(cq0.pop_front());
      end
      c_rdata_valid <= '0;
      if (rq.size() != 0 && rq[0].t <= cyc) begin
        c_rdata_valid[rq[0].p] <= 1'b1;
        c_rdata_data[rq[0].p] <= rq[0].d;
        n_ret++;
        if (n_ret == dual) c_rdata_valid[!rq[0].p] <= 1'b1;
        void'(rq.pop_front());
      end
      c_cmd_ready[0] <= cq0.size() < QD && ($urandom % 100) >= stall;
      c_cmd_ready[1] <= cq1.size() < QD && ($urandom % 100) >= stall;
    end
    cyc++;
  end

  string dir;
  integer fd, nread;
  logic [31:0] words [BEATS * 16];
  initial begin
    void'($value$plusargs("axi_stall=%d", stall));
    void'($value$plusargs("axi_lat=%d", lat));
    void'($value$plusargs("ldn_busy=%d", busy));
    void'($value$plusargs("ldn_dual=%d", dual));
    for (int i = 0; i < BEATS; i++) mem[i] = '0;
    if (IMG && $value$plusargs("dir=%s", dir)) begin
      for (int i = 0; i < BEATS * 16; i++) words[i] = '0;
      fd = $fopen($sformatf("%s/ch%0d.bin", dir, CH), "rb");
      if (fd != 0) begin
        nread = $fread(words, fd);
        $fclose(fd);
      end
      for (int i = 0; i < BEATS * 16; i++) mem[i / 16][32 * (i % 16) +: 32] = words[i];
    end
  end
  // dump may span several of this clock's cycles (it is driven in the core clock): its first
  logic dump_q = 1'b0;
  always @(posedge clk) dump_q <= dump;
  always @(posedge clk) if (dump && !dump_q) begin
    $display("LDN ch%0d rd=%0d wr=%0d busy=%0d", CH, n_rd, n_wr, n_busy);
    if (IMG) begin
      fd = $fopen($sformatf("%s/ch%0d_out.bin", dir, CH), "wb");
      for (int i = 0; i < BEATS * 16; i++) $fwrite(fd, "%u", mem[i / 16][32 * (i % 16) +: 32]);
      $fclose(fd);
    end
  end
endmodule
