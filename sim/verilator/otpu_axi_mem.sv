// Simulation model of the board memory: two AXI4 slave channels (512-bit, single-beat
// transactions) in front of one logical DRAM image with the 64-byte channel interleave of
// otpu_axi_dram. Every ready is randomly withheld and every response randomly delayed (seed
// +axi_seed=N, stall probability +axi_stall=percent), so the slice sees variable latency and
// backpressure; reads and writes are not ordered against each other, as in a real controller.
// Bandwidth: +axi_bw=P limits each channel to P percent of one 64-byte beat per cycle (reads and
// writes together; 100 = no limit); +axi_lat=N overrides the minimum latency.
// Images load from dram_<SID>.bin and dump to dram_out_<SID>.bin, as otpu_dram. With PHYS = 1
// the files are the channels' own memories instead, as the host sees them: ch<c>.bin (big-endian
// words, as $fread reads) in, ch<c>_out.bin (little-endian) out, WORDS / 2 words each.
module otpu_axi_mem #(
  parameter int WORDS = 1 << 18,
  parameter int PHYS  = 0,
  parameter int LAT   = 20,              // minimum read / write-response latency
  parameter int SID   = 0,
  parameter logic [31:0] BASE0 = 32'h0000_0000,
  parameter logic [31:0] BASE1 = 32'h8000_0000
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic [1:0]            s_awvalid,
  output logic [1:0]            s_awready,
  input  logic [1:0][31:0]      s_awaddr,
  input  logic [1:0]            s_awid,
  input  logic [1:0]            s_wvalid,
  output logic [1:0]            s_wready,
  input  logic [1:0][511:0]     s_wdata,
  input  logic [1:0][63:0]      s_wstrb,
  output logic [1:0]            s_bvalid,
  input  logic [1:0]            s_bready,
  output logic [1:0]            s_bid,
  output logic [1:0][1:0]       s_bresp,
  input  logic [1:0]            s_arvalid,
  output logic [1:0]            s_arready,
  input  logic [1:0][31:0]      s_araddr,
  input  logic [1:0]            s_arid,
  output logic [1:0]            s_rvalid,
  input  logic [1:0]            s_rready,
  output logic [1:0]            s_rid,
  output logic [1:0][511:0]     s_rdata,
  output logic [1:0][1:0]       s_rresp,
  output logic [1:0]            s_rlast,
  input  logic                  dump
);
  logic [31:0] mem [WORDS];
  int stall = 0;
  int bw = 100;
  int lat = LAT;
  longint cyc = 0;

  function automatic int beat_word(input logic [31:0] addr, input int c);
    logic [31:0] off;
    off = addr - (c ? BASE1 : BASE0);
    return int'(((off >> 6) * 2 + c) * 16);
  endfunction
  function automatic bit rnd_stall();
    return ($urandom % 100) < stall;
  endfunction

  typedef struct { longint t; logic id; logic [31:0] addr; } rq_t;
  typedef struct { longint t; logic id; } bq_t;
  rq_t rq [2][$];
  bq_t bq [2][$];
  logic [31:0] aw_a [2][$];
  logic        aw_i [2][$];
  logic [511:0] w_d [2][$];
  logic [63:0]  w_s [2][$];

  always_ff @(posedge clk) cyc <= cyc + 1;

  for (genvar c = 0; c < 2; c++) begin : g_ch
    logic arr, awr, wr, rv, bv;
    int cr = 0;                          // bandwidth credit (100 = one beat)
    always_ff @(posedge clk) begin
      arr <= !rnd_stall();
      awr <= !rnd_stall();
      wr  <= !rnd_stall();
    end
    assign s_arready[c] = arr;
    assign s_awready[c] = awr;
    assign s_wready[c] = wr;
    // R: the head read, once its time has come (in order per channel)
    always_comb begin
      s_rvalid[c] = 1'b0; s_rid[c] = 1'b0; s_rdata[c] = '0;
      s_rresp[c] = 2'b00; s_rlast[c] = 1'b1;
      if (rv) begin
        s_rvalid[c] = 1'b1;
        s_rid[c] = rq[c][0].id;
        for (int k = 0; k < 16; k++) s_rdata[c][32 * k +: 32] = mem[beat_word(rq[c][0].addr, c) + k];
      end
      s_bvalid[c] = bv;
      s_bid[c] = bv ? bq[c][0].id : 1'b0;
      s_bresp[c] = 2'b00;
    end
    always_ff @(posedge clk) begin
      if (rst) begin
        rq[c].delete(); bq[c].delete(); aw_a[c].delete(); aw_i[c].delete();
        w_d[c].delete(); w_s[c].delete();
        rv <= 1'b0; bv <= 1'b0;
      end else begin
        if (s_arvalid[c] && s_arready[c]) begin
          if (beat_word(s_araddr[c], c) + 16 > WORDS) $fatal(1, "AXI read beyond memory");
          rq[c].push_back('{cyc + lat + ($urandom % 8), s_arid[c], s_araddr[c]});
        end
        if (s_awvalid[c] && s_awready[c]) begin
          aw_a[c].push_back(s_awaddr[c]);
          aw_i[c].push_back(s_awid[c]);
        end
        if (s_wvalid[c] && s_wready[c]) begin
          w_d[c].push_back(s_wdata[c]);
          w_s[c].push_back(s_wstrb[c]);
        end
        cr = (cr + bw > 200) ? 200 : cr + bw;
        // a write is performed once both its address and data are in (and the channel has time)
        if (aw_a[c].size() != 0 && w_d[c].size() != 0 && cr >= 100) begin
          int b;
          cr = cr - 100;
          b = beat_word(aw_a[c][0], c);
          if (b + 16 > WORDS) $fatal(1, "AXI write beyond memory");
          for (int k = 0; k < 64; k++)
            if (w_s[c][0][k]) mem[b + k / 4][8 * (k % 4) +: 8] <= w_d[c][0][8 * k +: 8];
          bq[c].push_back('{cyc + lat + ($urandom % 8), aw_i[c][0]});
          void'(aw_a[c].pop_front()); void'(aw_i[c].pop_front());
          void'(w_d[c].pop_front()); void'(w_s[c].pop_front());
        end
        if (rv && s_rready[c]) void'(rq[c].pop_front());
        if (bv && s_bready[c]) void'(bq[c].pop_front());
        // next cycle's responses (the queues above are already updated)
        if (rq[c].size() != 0 && rq[c][0].t <= cyc && !rnd_stall() && cr >= 100) begin
          rv <= 1'b1;
          cr = cr - 100;
        end else begin
          rv <= 1'b0;
        end
        bv <= bq[c].size() != 0 && bq[c][0].t <= cyc && !rnd_stall();
      end
    end
  end

  string dir;
  integer fd, nread;
  logic [31:0] chm [WORDS / 2];
  initial begin
    void'($value$plusargs("axi_stall=%d", stall));
    void'($value$plusargs("axi_bw=%d", bw));
    void'($value$plusargs("axi_lat=%d", lat));
    begin
      int seed;
      if ($value$plusargs("axi_seed=%d", seed)) void'($urandom(seed));
    end
    for (int i = 0; i < WORDS; i++) mem[i] = '0;
    if ($value$plusargs("dir=%s", dir)) begin
      if (PHYS == 0) begin
        fd = $fopen($sformatf("%s/dram_%0d.bin", dir, SID), "rb");
        if (fd != 0) begin
          nread = $fread(mem, fd);
          $fclose(fd);
        end
      end else begin
        // channel c word j (beat j / 16) is logical beat 2 * (j / 16) + c
        for (int c = 0; c < 2; c++) begin
          for (int i = 0; i < WORDS / 2; i++) chm[i] = '0;
          fd = $fopen($sformatf("%s/ch%0d.bin", dir, c), "rb");
          if (fd != 0) begin
            nread = $fread(chm, fd);
            $fclose(fd);
          end
          for (int j = 0; j < WORDS / 2; j++) mem[(2 * (j / 16) + c) * 16 + j % 16] = chm[j];
        end
      end
    end
  end
  always @(posedge clk) if (dump) begin
    if (PHYS == 0) begin
      fd = $fopen($sformatf("%s/dram_out_%0d.bin", dir, SID), "wb");
      for (int i = 0; i < WORDS; i++) $fwrite(fd, "%u", mem[i]);
      $fclose(fd);
    end else begin
      for (int c = 0; c < 2; c++) begin
        fd = $fopen($sformatf("%s/ch%0d_out.bin", dir, c), "wb");
        for (int j = 0; j < WORDS / 2; j++) $fwrite(fd, "%u", mem[(2 * (j / 16) + c) * 16 + j % 16]);
        $fclose(fd);
      end
    end
  end
endmodule
