// Simulation model of one LiteDRAM native port as LiteDRAMCrossbar serves a single master (64-byte
// beats: 576 bits with ECC, 512 to the user; beat addresses), as otpu_mem_ch drives it.
// cmd ready is randomly withheld (+axi_stall=percent) and drops while QD commands are held (the
// bank's command buffer and pipeline). Commands are performed in command order, at most one per
// cycle, each 1 to WJ cycles after it was taken or later (default QD 12, WJ 12), never during a
// busy stretch (refresh, row changes: +ldn_busy=N, a 10-49 cycle pause with probability N per
// mille per cycle). A write is performed by asking for its data (wdata ready for one cycle) and
// taking whatever is on wdata, as the crossbar does, without looking at valid: data not valid then
// is an error, and so is any byte not written (LiteDRAMNativePortECC rejects partial writes). A
// read returns the memory as it is when it is performed, LAT to LAT + 7 cycles later (+axi_lat=N),
// in command order, never held back.
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
  input  logic         clk,
  input  logic         rst,
  input  logic         c_cmd_valid,
  output logic         c_cmd_ready,
  input  logic         c_cmd_we,
  input  logic [24:0]  c_cmd_addr,
  input  logic         c_wdata_valid,
  output logic         c_wdata_ready,
  input  logic [511:0] c_wdata_data,
  input  logic [63:0]  c_wdata_we,
  output logic         c_rdata_valid,
  output logic [511:0] c_rdata_data,
  input  logic         dump
);
  logic [511:0] mem [BEATS];
  int stall = 0, lat = LAT, busy = 2;
  longint cyc = 0, n_rd = 0, n_wr = 0, n_busy = 0, texec = 0, tret = 0, busy_until = 0;
  typedef struct { longint t; logic we; logic [24:0] a; } c_t;
  typedef struct { longint t; logic [511:0] d; } r_t;
  c_t cq [$];
  r_t rq [$];
  logic [24:0] wa;                          // the write whose data is asked for this cycle

  always @(posedge clk) begin
    if (rst) begin
      cq.delete(); rq.delete();
      c_cmd_ready <= 1'b0; c_wdata_ready <= 1'b0; c_rdata_valid <= 1'b0;
      texec = cyc; tret = cyc; busy_until = cyc;
    end else begin
      // the write data asked for in the cycle now ending
      if (c_wdata_ready) begin
        if (!c_wdata_valid) $fatal(1, "otpu_ldn_model ch%0d: write data asked for, not valid", CH);
        if (c_wdata_we != '1) $fatal(1, "otpu_ldn_model ch%0d: partial write (we %h)", CH, c_wdata_we);
        mem[wa] = c_wdata_data;
        n_wr++;
      end
      // a command taken
      if (c_cmd_valid && c_cmd_ready) begin
        longint t;
        if (c_cmd_addr >= BEATS) $fatal(1, "otpu_ldn_model ch%0d: beat %0d beyond the memory", CH, c_cmd_addr);
        t = cyc + 1 + longint'($urandom % WJ);
        if (t <= texec) t = texec + 1;
        texec = t;
        cq.push_back('{t, c_cmd_we, c_cmd_addr});
      end
      // the head command performed
      c_wdata_ready <= 1'b0;
      if (busy != 0 && cyc >= busy_until && ($urandom % 1000) < busy) begin
        busy_until = cyc + 10 + longint'($urandom % 40);
        n_busy++;
      end
      if (cq.size() != 0 && cq[0].t <= cyc && cyc >= busy_until) begin
        if (cq[0].we) begin
          c_wdata_ready <= 1'b1;
          wa = cq[0].a;
        end else begin
          longint t;
          t = cyc + lat + longint'($urandom % 8);
          if (t <= tret) t = tret + 1;
          tret = t;
          rq.push_back('{t, mem[cq[0].a]});
          n_rd++;
        end
        void'(cq.pop_front());
      end
      if (rq.size() != 0 && rq[0].t <= cyc) begin
        c_rdata_valid <= 1'b1;
        c_rdata_data <= rq[0].d;
        void'(rq.pop_front());
      end else begin
        c_rdata_valid <= 1'b0;
      end
      c_cmd_ready <= cq.size() < QD && ($urandom % 100) >= stall;
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
